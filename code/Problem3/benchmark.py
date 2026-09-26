"""独立的第三问实验：结构覆盖、1～5 核同方案双配置对照及文件级官方核验。"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import statistics
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.graph import Graph
from tools.plan import Plan
from tools.official import CacheSettings, ROOT, REFERENCE, read_json, write_json
from tools.experiment import digest, write_csv
from Problem3.solve import Solver, Solution, save_verified
from Problem3.diagnostics import analyze, compact

DEFAULT_CASES = (1, 2, 4, 8, 10, 14, 16, 20, 25, 29, 37, 44, 51, 58, 62, 67, 69, 84, 85, 93)


def config_matches(manifest, path):
    """允许 Windows 检出的换行差异，同时逐值核对全部物理参数。"""
    normalized = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    return (manifest['config_sha256'] in (digest(path), normalized)
            and manifest['settings'] == vars(CacheSettings.read(path)))


def features(raw):
    graph = Graph(raw)
    value = graph.features()
    inputs = [t for t in graph.tensors if graph.consumers[t] and not graph.producers[t]]
    value.update(input_bytes=sum(graph.tensors[t]['size'] for t in inputs),
                 shared_inputs=sum(len(graph.consumers[t]) > 1 for t in inputs))
    return value


def source_hashes():
    paths = list((ROOT / 'code/tools').glob('*.py')) + list(Path(__file__).parent.glob('*.py'))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def run_case(job):
    path, config, output, options, refine_options, previous = job
    started = time.perf_counter()
    solver = Solver(read_json(path), CacheSettings.read(config))
    folder = output / path.stem
    whole, whole_without = solver.whole
    write_json(folder / 'baseline.json', dict(l2=compact(whole.result), no_l2=compact(whole_without),
                                             source='fresh_official_whole_graph'))
    setup = time.perf_counter() - started
    info, rows = solver.graph.features(), []
    for n in range(1, 6):
        solve_started = time.perf_counter()
        if previous:
            source = previous / path.stem / f'core_{n}'
            plan = Plan.read(solver.graph, read_json(source / f'{path.stem}_multicore_res.json'))
            if len(plan.blocks) != n:
                raise ValueError(f'{path.stem}: 初始方案核数不符')
            result = solver.settings.evaluate(solver.graph.raw, plan.as_dict())
            without = solver.settings.without_cache(solver.graph.raw, plan.as_dict())
            recorded = read_json(source / 'official.json')
            if (json.loads(json.dumps(compact(result))) != recorded['l2'] or
                    json.loads(json.dumps(compact(without))) != recorded['no_l2']):
                raise ValueError(f'{path.stem}/{n}: 历史方案与当前官方双配置重评不一致')
            solution = Solution(plan, result, 'initial_plan')
            report = read_json(source / 'search.json')
        else:
            solution, report = solver.solve(n, **options)
        solution, refinement = solver.refine(solution, **refine_options)
        directory = folder / f'core_{n}'
        plan_path = directory / f'{path.stem}_multicore_res.json'
        result, without, verified = save_verified(solver, solution, plan_path)
        diag = analyze(result, solver.settings)
        write_json(directory / 'official.json', dict(l2=compact(result), no_l2=compact(without), **verified))
        write_json(directory / 'search.json', report)
        write_json(directory / 'refinement.json', refinement)
        write_json(directory / 'diagnostics.json', diag)
        movement, stats = result['data_movement_bytes'], result['cache_stats']
        t, no_l2, single = result['makespan'], without['makespan'], whole_without['makespan']
        row = dict(case=path.stem, family=info['family'], compute_ops=info['compute_ops'], cores=n,
                   makespan=t, no_l2_makespan=no_l2, whole_no_l2_makespan=single,
                   whole_l2_makespan=whole.score[0], cache_speedup=no_l2 / max(1, t),
                   speedup_l2=single / max(1, t), speedup_no_l2=single / max(1, no_l2),
                   normalized_l2=t / max(1, single), normalized_no_l2=no_l2 / max(1, single),
                   hit_rate=stats['hit_rate'], hits=stats['hits'], misses=stats['copy_in_misses'],
                   hit_bytes=stats['hit_bytes'], miss_bytes=stats['miss_bytes'],
                   added_bytes=movement['added_copy_bytes'], scheduled_bytes=movement['scheduled_copy_bytes'],
                   spill_bytes=movement['spill_added_copy_bytes'],
                   partition_added_bytes=movement['partition_added_copy_bytes'],
                   no_l2_added_bytes=without['data_movement_bytes']['added_copy_bytes'],
                   subgraphs=sum(map(len, solution.plan.blocks)), active_cores=sum(bool(c) for c in solution.plan.blocks),
                   peak_L1=max((p['L1'] for p in result['memory_peak_by_core'].values()), default=0),
                   peak_UB=max((p['UB'] for p in result['memory_peak_by_core'].values()), default=0),
                   ddr_demand_fraction=diag['ddr_byte_demand_fraction'],
                   cache_demand_fraction=diag['cache_byte_demand_fraction'],
                   compute_pipe_fraction=diag['max_compute_pipe_fraction'],
                   cold_duplicate_misses=diag['cold_duplicate_misses'], post_fill_misses=diag['post_fill_misses'],
                   cross_transfers=diag['cross_transfers'], max_copy_queue_wait=diag['max_copy_queue_wait'],
                   carry_makespan=report['carry_makespan'], initial_makespan=report['initial_makespan'],
                   structural_makespan=report['structural_makespan'],
                   cache_makespan=report['cache_makespan'],
                   lower_bound=report['global_lower_bound'], gap_upper_bound=report['gap_upper_bound'],
                   certified_optimal=report['certified_optimal'],
                   search_seconds=report['search_seconds'], verification_seconds=verified['verification_seconds'],
                   solve_seconds=time.perf_counter() - solve_started, shared_setup_seconds=setup,
                   preview_calls=report['preview_calls'], rejected=report['rejected'], pruned=report['pruned'],
                   preparation_hits=report['preparation_hits'], preparation_misses=report['preparation_misses'],
                   simulation_hits=report['simulation_hits'],
                   budget_exhausted=report['budget_exhausted'], selected=solution.origin,
                   reference_verified=bool(previous),
                   before_refine_makespan=refinement['baseline_makespan'],
                   before_refine_no_l2=refinement['baseline_no_l2_makespan'],
                   before_refine_added_bytes=refinement['baseline_added_bytes'],
                   refine_candidates=refinement['candidates'],
                   refine_seconds=refinement['search_seconds'],
                   plan_sha256=verified['plan_sha256'], verified=True)
        rows.append(row)
        write_json(folder / 'rows.json', rows)
        print(f'{path.stem} N={n}: T3={t}, T2={no_l2}, hit={stats["hit_rate"]:.4f}, '
              f'{solution.origin}, search={report["search_seconds"]:.2f}s, verified', flush=True)
    del solver
    gc.collect()
    write_json(folder / 'completed.json', dict(seconds=time.perf_counter() - started,
                                              input_sha256=digest(path), features=info))
    return rows


def summarize(rows):
    curves = []
    cases = {r['case'] for r in rows}
    for n in range(1, 6):
        values = [r for r in rows if r['cores'] == n]
        if len(values) != len(cases) or {r['case'] for r in values} != cases:
            raise ValueError('对比曲线要求每种核数覆盖相同、完整且无重复的用例集合')
        mean = lambda key: statistics.mean(r[key] for r in values)
        curves.append(dict(cores=n, cases=len(values), mean_no_l2_normalized=mean('normalized_no_l2'),
                           mean_l2_normalized=mean('normalized_l2'), mean_no_l2_speedup=mean('speedup_no_l2'),
                           mean_l2_speedup=mean('speedup_l2'), mean_cache_speedup=mean('cache_speedup'),
                           median_cache_speedup=statistics.median(r['cache_speedup'] for r in values),
                           mean_hit_rate=mean('hit_rate'),
                           weighted_hit_rate=sum(r['hit_bytes'] for r in values) /
                           max(1, sum(r['hit_bytes'] + r['miss_bytes'] for r in values)),
                           cache_wins=sum(r['makespan'] < r['no_l2_makespan'] for r in values),
                           cache_losses=sum(r['makespan'] > r['no_l2_makespan'] for r in values),
                           structure_wins=sum(r['structural_makespan'] < r['initial_makespan'] for r in values),
                           cache_search_wins=sum(r['cache_makespan'] < r['structural_makespan'] for r in values),
                           polish_wins=sum(r['makespan'] < r['cache_makespan'] for r in values),
                           mean_search_seconds=mean('search_seconds'), mean_solve_seconds=mean('solve_seconds'),
                           budget_exhausted=sum(r['budget_exhausted'] for r in values)))
    for case in cases:
        values = sorted((r for r in rows if r['case'] == case), key=lambda r: r['cores'])
        scores = [(r.get('before_refine_makespan', r['makespan']),
                   r.get('before_refine_added_bytes', r['added_bytes'])) for r in values]
        if any(b > a for a, b in zip(scores, scores[1:])):
            raise AssertionError(f'{case}: 增加可用核心后的保底方案丢失')
    return curves


def compare(rows, previous, manifest):
    """核对相同输入、配置与评测标准；旧散列缺失时必须逐方案双配置重评。"""
    old_manifest = read_json(previous / 'manifest.json')
    for key in ('config_sha256', 'reference_sha256'):
        if old_manifest[key] != manifest[key]:
            if key == 'config_sha256' and config_matches(old_manifest, Path(manifest['config'])):
                continue
            verified_missing_hash = (key == 'reference_sha256' and not old_manifest[key]
                and manifest.get('refine_from') == str(previous.resolve())
                and all(r.get('reference_verified') for r in rows))
            if not verified_missing_hash:
                raise ValueError(f'两次实验的 {key} 不同，不能作为同标准对照')
    with (previous / 'metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        old = {(r['case'], int(r['cores'])): r for r in csv.DictReader(stream)}
    comparisons = []
    for row in rows:
        case, n = row['case'], row['cores']
        if old_manifest['inputs'].get(case) != manifest['inputs'][case]:
            raise ValueError(f'{case}: 原始图不同')
        before = old[case, n]
        if int(before['whole_no_l2_makespan']) != row['whole_no_l2_makespan']:
            raise ValueError(f'{case}: 原图单核基准不同')
        t, a = int(before['makespan']), int(before['added_bytes'])
        comparisons.append(dict(case=case, cores=n, old_makespan=t, new_makespan=row['makespan'],
            time_gain_pct=100 * (t - row['makespan']) / max(1, t),
            old_added_bytes=a, new_added_bytes=row['added_bytes'],
            score_nonregression=(row['makespan'], row['added_bytes']) <= (t, a),
            old_speedup=float(before['speedup_l2']), new_speedup=row['speedup_l2'],
            old_cache_speedup=float(before['cache_speedup']), new_cache_speedup=row['cache_speedup'],
            old_no_l2_makespan=int(before['no_l2_makespan']), new_no_l2_makespan=row['no_l2_makespan'],
            all_metrics_nonregression=(row['makespan'] <= t and row['added_bytes'] <= a
                and row['no_l2_makespan'] <= int(before['no_l2_makespan'])
                and row['no_l2_makespan'] * t >= int(before['no_l2_makespan']) * row['makespan']),
            old_search_seconds=float(before['search_seconds']), new_search_seconds=row['search_seconds']))
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=REFERENCE.parent / 'data')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=int, nargs='+', default=DEFAULT_CASES)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--budget', type=float, default=45)
    parser.add_argument('--seeds', type=int, default=18)
    parser.add_argument('--local', type=int, default=32)
    parser.add_argument('--cache', type=int, default=16)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--refine-budget', type=float, default=15)
    parser.add_argument('--refine-candidates', type=int, default=64)
    parser.add_argument('--refine-from', type=Path, help='对已保存的第三问方案做同源核验和非退化追加优化')
    parser.add_argument('--compare', type=Path, help='独立求解后与此目录的历史指标对照')
    args = parser.parse_args()
    if args.workers < 1 or len(set(args.cases)) != len(args.cases):
        parser.error('workers 必须为正整数；用例编号不得重复')
    if not math.isfinite(args.budget) or args.budget <= 0 or min(args.seeds, args.local, args.cache, args.rounds) < 0:
        parser.error('预算须为有限正数；搜索数量须为非负整数')
    if not math.isfinite(args.refine_budget) or args.refine_budget < 0 or args.refine_candidates < 0:
        parser.error('追加时间与候选数量须为非负数')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('输出目录必须为空，避免覆盖其他实验')
    selected = [args.data / f'case_{i:03d}.json' for i in sorted(args.cases)]
    if any(not p.is_file() for p in selected):
        parser.error('存在找不到的用例文件')
    config = args.config or args.data / 'config.txt'
    if args.refine_from:
        source_manifest = read_json(args.refine_from / 'manifest.json')
        if not config_matches(source_manifest, config):
            parser.error('初始方案的配置与当前配置不同')
        if any(source_manifest['inputs'].get(p.stem) != digest(p) for p in selected):
            parser.error('初始方案的输入图与当前输入图不同')
        args.compare = args.compare or args.refine_from
    inventory = {p.stem: features(read_json(p)) for p in sorted(args.data.glob('case_*.json'))}
    options = dict(budget=args.budget, seeds=args.seeds, local=args.local, cache=args.cache, rounds=args.rounds)
    refine_options = dict(budget=args.refine_budget, candidates=args.refine_candidates)
    manifest = dict(version='3.4', independent=not bool(args.refine_from), python=sys.version, platform=platform.platform(),
                    config=str(config.resolve()), config_sha256=digest(config), settings=vars(CacheSettings.read(config)),
                    source_sha256=source_hashes(), reference_sha256={p.name: digest(p) for p in REFERENCE.glob('*.py')},
                    inputs={p.stem: digest(p) for p in selected},
                    selected={p.stem: inventory[p.stem] for p in selected},
                    all_families=sorted({x['family'] for x in inventory.values()}),
                    covered_families=sorted({inventory[p.stem]['family'] for p in selected}),
                    search=options, refinement=refine_options, workers=args.workers,
                    refine_from=str(args.refine_from.resolve()) if args.refine_from else None,
                    comparison='same submitted plan, original problem_2/problem_3 evaluators')
    write_json(args.output / 'manifest.json', manifest)
    started, rows, failures = time.perf_counter(), [], {}
    jobs = [(p, config, args.output, options, refine_options, args.refine_from) for p in
            sorted(selected, key=lambda p: -inventory[p.stem]['compute_ops'])]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, job): job[0].stem for job in jobs}
        for future in as_completed(futures):
            case = futures[future]
            try:
                rows.extend(future.result())
            except Exception as error:
                failures[case] = f'{type(error).__name__}: {error}'
                checkpoint = args.output / case / 'rows.json'
                if checkpoint.exists():
                    rows.extend(read_json(checkpoint))
                print(f'{case}: FAILED: {failures[case]}', flush=True)
    rows.sort(key=lambda r: (r['case'], r['cores']))
    write_csv(args.output / 'metrics.csv', rows)
    complete = len(rows) == len(selected) * 5 and not failures
    if source_hashes() != manifest['source_sha256'] or digest(config) != manifest['config_sha256']:
        raise RuntimeError('实验运行期间代码或配置被修改，本次结果不能作为固定版本报告')
    if ({p.stem: digest(p) for p in selected} != manifest['inputs'] or
            {p.name: digest(p) for p in REFERENCE.glob('*.py')} != manifest['reference_sha256']):
        raise RuntimeError('实验运行期间原始输入或官方源码被修改')
    curves = summarize(rows) if complete else []
    write_csv(args.output / 'curves.csv', curves)
    summary = dict(wall_seconds=time.perf_counter() - started, completed=complete,
                   requested_cases=len(selected), rows=len(rows), failures=failures,
                   all_families_covered=manifest['all_families'] == manifest['covered_families'],
                   search_seconds=sum(r['search_seconds'] for r in rows),
                   verification_seconds=sum(r['verification_seconds'] for r in rows),
                   preview_calls=sum(r['preview_calls'] for r in rows),
                   official_final_calls=2 * len(rows), curves=curves)
    if complete and args.compare:
        comparisons = compare(rows, args.compare, manifest)
        write_csv(args.output / 'comparison.csv', comparisons)
        summary['previous_version'] = dict(path=str(args.compare.resolve()),
            metrics_sha256=digest(args.compare / 'metrics.csv'),
            wins=sum(r['new_makespan'] < r['old_makespan'] for r in comparisons),
            ties=sum(r['new_makespan'] == r['old_makespan'] for r in comparisons),
            losses=sum(r['new_makespan'] > r['old_makespan'] for r in comparisons),
            score_nonregressions=sum(r['score_nonregression'] for r in comparisons),
            all_metrics_nonregressions=sum(r['all_metrics_nonregression'] for r in comparisons),
            old_mean_speedup=statistics.mean(r['old_speedup'] for r in comparisons),
            new_mean_speedup=statistics.mean(r['new_speedup'] for r in comparisons),
            old_mean_cache_speedup=statistics.mean(r['old_cache_speedup'] for r in comparisons),
            new_mean_cache_speedup=statistics.mean(r['new_cache_speedup'] for r in comparisons),
            old_search_seconds=sum(r['old_search_seconds'] for r in comparisons),
            new_search_seconds=sum(r['new_search_seconds'] for r in comparisons))
    write_json(args.output / 'summary.json', summary)
    print(f'completed={complete}; rows={len(rows)}; wall={summary["wall_seconds"]:.2f}s; failures={failures}', flush=True)
    if not complete:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

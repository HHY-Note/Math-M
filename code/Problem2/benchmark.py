"""场景 B 批量实验：同组用例的 2～5 核官方结果、阶段对照及可追溯记录。"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import csv
import platform
import statistics
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.graph import Graph
from tools.experiment import digest, official_summary, write_csv
from tools.official import REFERENCE, ROOT, SceneBSettings, read_json, write_json
from Problem2.solve import Solver


def diagnostics(result, settings):
    """利用已评分的真实时间线记录瓶颈，不额外运行近似模拟器。"""
    timeline, finish, busy = result['per_core_timeline'], [], []
    for core in timeline:
        finish.append(max((op['end'] for op in core['ops']), default=0))
        busy.append({pipe: sum(op['duration'] for op in core['ops'] if op['pipe'] == pipe)
                     for pipe in ('PIPE_M', 'PIPE_V')})
    waits = [item['copy_in_start'] - item['copy_in_release']
             for item in result['cross_core_transfers']]
    t = result['makespan']
    return dict(core_finish=finish, pipe_busy=busy, cross_transfers=len(waits),
                mean_copy_queue_wait=statistics.mean(waits) if waits else 0,
                max_copy_queue_wait=max(waits, default=0),
                # 字节/B 是理想连续带宽需求；未计入每条 COPY 的整数取整。
                ddr_byte_demand_fraction=result['data_movement_bytes']['scheduled_copy_bytes']
                / settings.bandwidth / t,
                max_compute_pipe_fraction=max((max(p.values()) for p in busy), default=0) / t)


def run_case(job):
    path, config, output, budget, candidates, orders, moves = job
    started = time.perf_counter()
    settings = SceneBSettings.read(config)
    solver = Solver(read_json(path), settings)
    folder = output / path.stem
    features = solver.graph.features()
    write_json(folder / 'singlecore.json', solver.single)
    setup = time.perf_counter() - started
    rows = []
    for n in range(2, 6):
        filename = path.stem + '_multicore_res.json'
        best, report = solver.solve(n, budget, candidates, orders, moves)
        directory = folder / f'core_{n}'
        write_json(directory / filename, best.plan.as_dict())
        write_json(directory / 'official.json', official_summary(best.result))
        write_json(directory / 'search.json', report)
        diag = diagnostics(best.result, settings)
        write_json(directory / 'diagnostics.json', diag)
        movement = best.result['data_movement_bytes']
        row = dict(case=path.stem, family=features['family'], compute_ops=features['compute_ops'],
                   cores=n, singlecore_makespan=solver.single['makespan'], makespan=best.score[0],
                   speedup=solver.single['makespan'] / best.score[0],
                   initial_makespan=report['initial_makespan'],
                   ordered_makespan=report['ordered_makespan'],
                   carry_makespan=report['carry_makespan'],
                   initial_speedup=solver.single['makespan'] / report['initial_makespan'],
                   added_bytes=movement['added_copy_bytes'],
                   partition_added_bytes=movement['partition_added_copy_bytes'],
                   spill_added_bytes=movement['spill_added_copy_bytes'],
                   scheduled_bytes=movement['scheduled_copy_bytes'],
                   subgraphs=sum(map(len, best.plan.blocks)),
                   active_cores=sum(bool(core) for core in best.plan.blocks),
                   peak_L1=max(p['L1'] for p in best.result['memory_peak_by_core'].values()),
                   peak_UB=max(p['UB'] for p in best.result['memory_peak_by_core'].values()),
                   cross_transfers=diag['cross_transfers'],
                   ddr_byte_demand_fraction=diag['ddr_byte_demand_fraction'],
                   max_compute_pipe_fraction=diag['max_compute_pipe_fraction'],
                   mean_copy_queue_wait=diag['mean_copy_queue_wait'],
                   max_copy_queue_wait=diag['max_copy_queue_wait'],
                   solve_seconds=report['elapsed_seconds'], shared_setup_seconds=setup,
                   search_seconds=report['search_seconds'], verification_seconds=report['verification_seconds'],
                   singlecore_source='fresh_official_whole_graph', independent=True,
                   official_calls=report['official_calls'], preview_calls=report['preview_calls'],
                   rejected=report['rejected'], pruned=report['pruned'],
                   global_lower_bound=report['global_lower_bound'], gap_upper_bound=report['gap_upper_bound'],
                   near_optimal=report['near_optimal'],
                   budget_exhausted=report['budget_exhausted'], selected=best.origin, verified=True)
        rows.append(row)
        write_json(folder / 'rows.json', rows)
        print(f'{path.stem} N={n}: T={best.score[0]}; speedup={row["speedup"]:.3f}; '
              f'{best.origin}; {report["elapsed_seconds"]:.2f}s', flush=True)
    elapsed = time.perf_counter() - started
    write_json(folder / 'completed.json', dict(seconds=elapsed, features=features, input_sha256=digest(path)))
    return rows


def compare(rows, folder, manifest):
    """所有新解生成结束后才读取旧指标；历史方案和成绩不进入求解进程。"""
    old_manifest = read_json(folder / 'manifest.json')
    for key in ('config_sha256', 'reference_sha256'):
        if old_manifest[key] != manifest[key]:
            raise ValueError(f'历史实验的 {key} 不一致，不能比较')
    with (folder / 'metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        previous = {(r['case'], int(r['cores'])): r for r in csv.DictReader(stream)}
    for row in rows:
        old = previous[row['case'], row['cores']]
        if (old_manifest['inputs'][row['case']] != manifest['inputs'][row['case']]
                or int(old['singlecore_makespan']) != row['singlecore_makespan']):
            raise ValueError(f'{row["case"]}: 输入或重新计算的单核基准不一致')
        row.update(previous_makespan=int(old['makespan']), previous_added_bytes=int(old['added_bytes']),
                   previous_speedup=float(old['speedup']), previous_solve_seconds=float(old['solve_seconds']))
        row['nonregression'] = ((row['makespan'], row['added_bytes'])
                               <= (row['previous_makespan'], row['previous_added_bytes']))
    return dict(folder=str(folder.resolve()), metrics_sha256=digest(folder / 'metrics.csv'),
                time_wins=sum(r['makespan'] < r['previous_makespan'] for r in rows),
                time_ties=sum(r['makespan'] == r['previous_makespan'] for r in rows),
                regressions=[dict(case=r['case'], cores=r['cores'], previous=r['previous_makespan'],
                                  current=r['makespan']) for r in rows if not r['nonregression']])


def summarize(rows):
    result = []
    cases = {r['case'] for r in rows}
    for n in range(2, 6):
        values = [r for r in rows if r['cores'] == n]
        if len(values) != len(cases) or {r['case'] for r in values} != cases:
            raise ValueError('各核数必须覆盖相同且无重复的用例集')
        result.append(dict(cores=n, cases=len(values),
                           mean_speedup=statistics.mean(r['speedup'] for r in values),
                           median_speedup=statistics.median(r['speedup'] for r in values),
                           initial_mean_speedup=statistics.mean(r['initial_speedup'] for r in values),
                           order_wins=sum(r['ordered_makespan'] < r['initial_makespan'] for r in values),
                           move_wins=sum(r['makespan'] < r['ordered_makespan'] for r in values),
                           mean_solve_seconds=statistics.mean(r['solve_seconds'] for r in values),
                           max_solve_seconds=max(r['solve_seconds'] for r in values)))
        if 'previous_speedup' in values[0]:
            result[-1]['previous_mean_speedup'] = statistics.mean(r['previous_speedup'] for r in values)
    for case in cases:
        values = sorted((r for r in rows if r['case'] == case), key=lambda r: r['cores'])
        scores = [(r['makespan'], r['added_bytes']) for r in values]
        if any(b > a for a, b in zip(scores, scores[1:])):
            raise AssertionError(f'{case}: 补空核保底未生效')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / '题目/data')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=int, nargs='+', required=True, help='明确指定本次用例编号')
    parser.add_argument('--compare', type=Path, help='求解全部结束后读取的历史实验指标目录')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--budget', type=float, default=45)
    parser.add_argument('--candidates', type=int, default=24)
    parser.add_argument('--orders', type=int, default=16)
    parser.add_argument('--moves', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or len(set(args.cases)) != len(args.cases):
        parser.error('workers 必须为正整数，用例编号不得重复')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('输出目录非空，请为本次实验指定新目录')
    config = args.config or args.data / 'config.txt'
    selected = [args.data / f'case_{i:03d}.json' for i in sorted(args.cases)]
    if any(not p.exists() for p in selected):
        parser.error('存在找不到的用例文件')
    inventory = {p.stem: Graph(read_json(p)).features() for p in sorted(args.data.glob('case_*.json'))}
    source = list((ROOT / 'code/tools').glob('*.py')) + list(Path(__file__).parent.glob('*.py'))
    manifest = dict(version='2.3', independent=True, python=sys.version, platform=platform.platform(),
                    config_sha256=digest(config), config=str(config.resolve()),
                    settings=vars(SceneBSettings.read(config)),
                    source_sha256={str(p.relative_to(ROOT)): digest(p) for p in source},
                    reference_sha256={p.name: digest(p) for p in REFERENCE.glob('*.py')},
                    inputs={p.stem: digest(p) for p in selected},
                    selected={p.stem: inventory[p.stem] for p in selected},
                    all_families=sorted({v['family'] for v in inventory.values()}),
                    covered_families=sorted({inventory[p.stem]['family'] for p in selected}),
                    search=dict(budget=args.budget, candidates=args.candidates,
                                orders=args.orders, moves=args.moves, workers=args.workers))
    write_json(args.output / 'manifest.json', manifest)
    started, rows, failures = time.perf_counter(), [], {}
    jobs = [(p, config, args.output, args.budget, args.candidates, args.orders, args.moves)
            for p in sorted(selected, key=lambda p: -inventory[p.stem]['compute_ops'])]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, job): job[0].stem for job in jobs}
        for future in as_completed(futures):
            case = futures[future]
            try:
                rows.extend(future.result())
            except Exception as error:
                failures[case] = f'{type(error).__name__}: {error}'
                print(f'{case}: FAILED: {failures[case]}', flush=True)
    rows.sort(key=lambda r: (r['case'], r['cores']))
    wall = time.perf_counter() - started
    comparison = compare(rows, args.compare, manifest) if args.compare and rows else None
    write_csv(args.output / 'metrics.csv', rows)
    summary = dict(wall_seconds=wall, requested_cases=len(selected), independent=True,
                   completed_cases=len({r['case'] for r in rows}), failures=failures,
                   comparison=comparison, solve_seconds=sum(r['solve_seconds'] for r in rows),
                   search_seconds=sum(r['search_seconds'] for r in rows),
                   verification_seconds=sum(r['verification_seconds'] for r in rows),
                   setup_seconds=sum(r['shared_setup_seconds'] for r in rows if r['cores'] == 2),
                   cores=summarize(rows) if rows else [])
    write_json(args.output / 'summary.json', summary)
    print(summary, flush=True)
    if failures or len(rows) != 4 * len(selected):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

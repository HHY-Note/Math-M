"""批量求解全部或指定用例，原评测核验单核及 2～5 核并记录结果。"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import csv
import json
import platform
import statistics
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.graph import Graph
from tools.experiment import digest, official_summary
from tools.official import REFERENCE, ROOT, Settings, read_json, write_json
from Problem1.execution import assert_official
from Problem1.solve import Solver

def verify(solution, raw, settings, previous=None, plan_path=None):
    """相同输入、配置、原评测器和方案复用已有核验；新方案逐操作核验。"""
    begin = time.perf_counter()
    if (previous is not None and previous.exists()
            and (plan_path is None or (plan_path.exists() and read_json(plan_path) == solution.plan))):
        summary = read_json(previous)
        for key in ('makespan', 'data_movement_bytes', 'memory_peak_by_core'):
            if summary[key] != json.loads(json.dumps(solution.result[key])):
                raise AssertionError(f'已有核验与本次模拟不一致: {key}')
        expected = {s: (c, solution.result['task_start'][s], solution.result['task_end'][s])
                    for c, order in enumerate(solution.schedules) for s in order}
        actual = {t['task_id']: (c['core_id'], t['start'], t['end'])
                  for c in summary['per_core_tasks'] for t in c['tasks']}
        if actual != expected or summary['num_cores'] != len(solution.schedules):
            raise AssertionError('已有核验与本次 Task 时间不一致')
        source = 'reused_identical_plan'
    else:
        reference = settings.evaluate(raw, solution.plan)
        assert_official(solution.compiled, solution.result, reference)
        summary, source = official_summary(reference), 'original_evaluator'
    return summary, time.perf_counter() - begin, source


def run_case(job):
    path, config, output, cores, budget, count, moves, compare = job
    started = time.perf_counter()
    raw = read_json(path)
    settings = Settings.read(config)
    solver = Solver(raw, settings)
    folder = Path(output) / Path(path).stem
    previous = Path(compare) / Path(path).stem if compare else None
    single = solver.singlecore()
    setup_seconds = time.perf_counter() - started
    reference, single_check, single_source = verify(
        single, raw, settings, previous / 'singlecore.json' if previous else None)
    write_json(folder / 'singlecore.json', reference)
    features = solver.graph.features()
    rows = []
    for n in cores:
        best, baseline, report = solver.solve(n, budget, count, moves)
        old_dir = previous / f'core_{n}' if previous else None
        plan_name = Path(path).stem + '_multicore_res.json'
        reference, check_seconds, source = verify(
            best, raw, settings, old_dir / 'official.json' if old_dir else None,
            old_dir / plan_name if old_dir else None)
        baseline_verified = False
        if n == 4:
            base_reference, _, _ = verify(
                baseline, raw, settings, previous / 'baseline_4core.json' if previous else None,
                previous / 'baseline_4core_plan.json' if previous else None)
            write_json(folder / 'baseline_4core.json', base_reference)
            write_json(folder / 'baseline_4core_plan.json', baseline.plan)
            baseline_verified = True
        directory = folder / f'core_{n}'
        write_json(directory / plan_name, best.plan)
        write_json(directory / 'official.json', reference)
        report['official_verified'] = True
        report['verification_source'] = source
        report['official_evaluation_seconds'] = check_seconds
        write_json(directory / 'search.json', report)
        result, movement = best.result, reference['data_movement_bytes']
        task_busy = [sum(result['task_end'][s] - result['task_start'][s] for s in order)
                     for order in best.schedules]
        row = dict(case=Path(path).stem, family=features['family'],
                   compute_ops=features['compute_ops'], cores=n,
                   singlecore_makespan=single.score[0], makespan=best.score[0],
                   speedup=single.score[0] / best.score[0],
                   baseline_makespan=baseline.score[0], baseline_added_bytes=baseline.score[1],
                   improvement_over_baseline=baseline.score[0] / best.score[0],
                   baseline_official_verified=baseline_verified,
                   added_bytes=movement['added_copy_bytes'],
                   partition_added_bytes=movement['partition_added_copy_bytes'],
                   spill_added_bytes=movement['spill_added_copy_bytes'],
                   scheduled_bytes=movement['scheduled_copy_bytes'],
                   tasks=len(best.compiled.partition.groups),
                   active_cores=sum(bool(order) for order in best.schedules),
                   ddr_service_fraction=result['ddr_full_bandwidth_work'] / best.score[0],
                   mean_task_occupancy=sum(task_busy) / (n * best.score[0]),
                   peak_L1=max(p['L1'] for p in result['memory_peak_by_core'].values()),
                   peak_UB=max(p['UB'] for p in result['memory_peak_by_core'].values()),
                   solve_seconds=report['elapsed_seconds'],
                   official_seconds=check_seconds, shared_setup_seconds=setup_seconds,
                   singlecore_official_seconds=single_check,
                   singlecore_verification_source=single_source, verification_source=source,
                   preparations=report['preparations'], simulations=report['simulations'],
                   selected=best.origin, verified=True)
        if old_dir:
            old_time = read_json(old_dir / 'official.json')['makespan']
            row.update(previous_makespan=old_time, previous_speedup=single.score[0] / old_time,
                       time_reduction=1 - best.score[0] / old_time)
        rows.append(row)
        write_json(folder / 'rows.json', rows)
        print(f'{Path(path).stem} N={n}: speedup={row["speedup"]:.3f}; '
              f'T={best.score[0]}; {source}', flush=True)
    elapsed = time.perf_counter() - started
    write_json(folder / 'completed.json', dict(case_seconds=elapsed, cores=cores,
                                              features=features, input_sha256=digest(path)))
    return rows, elapsed


def main():
    parser = argparse.ArgumentParser(description='问题一批量求解与完整评测')
    parser.add_argument('--data', type=Path, default=ROOT / '题目/data')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/Problem1/current')
    parser.add_argument('--compare', type=Path, help='已完整核验的旧实验目录，作为对照并复用相同方案核验')
    parser.add_argument('--cases', type=int, nargs='+', help='指定用例编号；省略时运行数据目录内全部用例')
    parser.add_argument('--cores', type=int, nargs='+', default=[2, 3, 4, 5])
    parser.add_argument('--budget', type=float, default=180)
    parser.add_argument('--max-candidates', type=int, default=3)
    parser.add_argument('--moves', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.compare and args.compare.resolve() == args.output.resolve():
        parser.error('输出目录不能覆盖对照数据')
    if args.cases is not None and len(set(args.cases)) != len(args.cases):
        parser.error('用例不能重复')
    if args.cores != sorted(set(args.cores)) or any(n not in range(2, 6) for n in args.cores):
        parser.error('cores 应为 2～5 中递增且不重复的核数')
    inventory = []
    for path in sorted(args.data.glob('case_[0-9][0-9][0-9].json')):
        features = Graph(read_json(path)).features()
        inventory.append(dict(case=path.stem, sha256=digest(path), **features))
    if not inventory:
        parser.error('数据目录中没有 case_编号.json 用例')
    names = ({f'case_{i:03}' for i in args.cases} if args.cases is not None
             else {row['case'] for row in inventory})
    selected = [row for row in inventory if row['case'] in names]
    if len(selected) != len(names):
        parser.error('部分用例不存在')
    families = sorted({row['family'] for row in inventory})
    covered = sorted({row['family'] for row in selected})
    source_paths = sorted((ROOT / 'code/Problem1').glob('*.py')) + sorted((ROOT / 'code/tools').glob('*.py'))
    manifest = dict(version='3.0', python=sys.version, platform=platform.platform(),
                    config_sha256=digest(args.data / 'config.txt'),
                    source_sha256={str(p.relative_to(ROOT)): digest(p) for p in source_paths},
                    reference_sha256={p.name: digest(p) for p in sorted(REFERENCE.glob('*.py'))},
                    all_families=families, covered_families=covered,
                    covers_all_observed_families=families == covered,
                    family_note='按算子集合及依赖深度划分的结构类别，不是题包提供的官方标签。',
                    selected=selected, settings=dict(cores=args.cores, budget=args.budget,
                                                     max_candidates=args.max_candidates,
                                                     moves=args.moves, workers=args.workers))
    if args.compare:
        previous = read_json(args.compare / 'manifest.json')
        for key in ('config_sha256', 'reference_sha256', 'python'):
            if manifest[key] != previous[key]:
                parser.error(f'不能复用已有核验，运行条件变化: {key}')
        known = {r['case']: r['sha256'] for r in previous['selected']}
        if any(known.get(r['case']) != r['sha256'] for r in selected):
            parser.error('对照用例缺失或输入文件发生变化')
        if any(not (args.compare / r['case'] / 'completed.json').exists() for r in selected):
            parser.error('对照用例没有完成原评测核验')
        manifest['comparison'] = str(args.compare.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'manifest.json', manifest)
    write_json(args.output / 'inventory.json', inventory)
    jobs = [(str(args.data / (row['case'] + '.json')), str(args.data / 'config.txt'),
             str(args.output), args.cores, args.budget, args.max_candidates, args.moves,
             str(args.compare) if args.compare else None)
            for row in sorted(selected, key=lambda r: -r['compute_ops'])]
    start, rows, case_times, errors = time.perf_counter(), [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_case = {executor.submit(run_case, job): Path(job[0]).stem for job in jobs}
        for future in as_completed(future_to_case):
            case = future_to_case[future]
            try:
                values, elapsed = future.result()
                rows.extend(values)
                case_times.append(dict(case=case, seconds=elapsed))
                print(f'COMPLETE {case} ({elapsed:.1f}s)', flush=True)
            except Exception as error:
                errors.append(dict(case=case, error=repr(error)))
                print(f'FAILED {case}: {error!r}', flush=True)
    rows.sort(key=lambda row: (row['case'], row['cores']))
    if rows:
        with (args.output / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(completed_cases=len(case_times), result_count=len(rows), errors=errors,
                   wall_seconds=time.perf_counter() - start, case_times=case_times,
                   by_cores={n: dict(
                       cases=sum(r['cores'] == n for r in rows),
                       mean_speedup=statistics.mean(r['speedup'] for r in rows if r['cores'] == n),
                       mean_improvement_over_baseline=statistics.mean(
                           r['improvement_over_baseline'] for r in rows if r['cores'] == n),
                       median_solve_seconds=statistics.median(
                           r['solve_seconds'] for r in rows if r['cores'] == n))
                       for n in args.cores if any(r['cores'] == n for r in rows)})
    if args.compare:
        for n, stats in summary['by_cores'].items():
            values = [r for r in rows if r['cores'] == n]
            stats.update(previous_mean_speedup=statistics.mean(r['previous_speedup'] for r in values),
                         mean_time_reduction=statistics.mean(r['time_reduction'] for r in values),
                         improved=sum(r['time_reduction'] > 0 for r in values),
                         regressed=sum(r['time_reduction'] < 0 for r in values))
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary['by_cores'], ensure_ascii=False), flush=True)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())

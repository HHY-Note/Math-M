"""问题一全量评测：100 个用例、每例 2～5 核，逐方案调用官方评测器。"""

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.experiment import digest, official_summary, write_csv
from tools.official import REFERENCE, ROOT, Settings, read_json, write_json
from Problem1.execution import assert_official
from Problem1.solve import Solver


CORES = (2, 3, 4, 5)


def verify(solution, raw, settings, plan_path=None):
    started = time.perf_counter()
    plan = read_json(plan_path) if plan_path else solution.plan
    if (not isinstance(plan, dict)
            or set(plan) != {'node_to_subgraph', 'core_schedules'}
            or plan != solution.plan):
        raise AssertionError('落盘方案与内存方案或规定的两字段格式不一致')
    reference = settings.evaluate(raw, plan)
    assert_official(solution.compiled, solution.result, reference)
    return official_summary(reference), time.perf_counter() - started


def run_case(job):
    path, config, output = map(Path, job)
    started = time.perf_counter()
    raw = read_json(path)
    settings = Settings.read(config)
    solver = Solver(raw, settings)
    folder = output / path.stem

    single = solver.singlecore()
    single_reference, single_check_seconds = verify(single, raw, settings)
    write_json(folder / 'singlecore.json', single_reference)
    features = solver.graph.features()

    rows = []
    for cores in CORES:
        best, baseline, report = solver.solve(cores)
        directory = folder / f'core_{cores}'
        plan_path = directory / f'{path.stem}_multicore_res.json'
        write_json(plan_path, best.plan)
        reference, check_seconds = verify(best, raw, settings, plan_path)
        write_json(directory / 'official.json', reference)
        write_json(directory / 'search.json', report)

        movement = reference['data_movement_bytes']
        result = best.result
        task_busy = sum(result['task_end'][task] - result['task_start'][task]
                        for order in best.schedules for task in order)
        rows.append(dict(
            case=path.stem, family=features['family'], compute_ops=features['compute_ops'],
            cores=cores, singlecore_makespan=single.score[0], makespan=best.score[0],
            speedup=single.score[0] / best.score[0],
            baseline_makespan=baseline.score[0],
            improvement_over_baseline=baseline.score[0] / best.score[0],
            added_bytes=movement['added_copy_bytes'],
            partition_added_bytes=movement['partition_added_copy_bytes'],
            spill_added_bytes=movement['spill_added_copy_bytes'],
            scheduled_bytes=movement['scheduled_copy_bytes'],
            tasks=len(best.compiled.partition.groups),
            active_cores=sum(bool(order) for order in best.schedules),
            ddr_service_fraction=result['ddr_full_bandwidth_work'] / best.score[0],
            mean_task_occupancy=task_busy / (cores * best.score[0]),
            peak_L1=max(peak['L1'] for peak in result['memory_peak_by_core'].values()),
            peak_UB=max(peak['UB'] for peak in result['memory_peak_by_core'].values()),
            solve_seconds=report['elapsed_seconds'], official_seconds=check_seconds,
            selected=best.origin, plan_sha256=digest(plan_path)))
        print(f'{path.stem} N={cores}: T={best.score[0]}, '
              f'speedup={rows[-1]["speedup"]:.3f}', flush=True)

    return rows, dict(case=path.stem, seconds=time.perf_counter() - started,
                      singlecore_official_seconds=single_check_seconds)


def main():
    parser = argparse.ArgumentParser(description='问题一：全量 100 用例、2～5 核评测')
    parser.add_argument('--data', type=Path, default=ROOT / '题目/data')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/Problem1_full')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error('workers 必须为正整数')
    paths = [args.data / f'case_{i:03}.json' for i in range(1, 101)]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        parser.error(f'缺少题目用例：{", ".join(missing)}')
    config = args.data / 'config.txt'
    if not config.is_file():
        parser.error(f'缺少配置文件：{config}')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('输出目录必须不存在或为空，以免混入旧结果')

    source_paths = sorted((ROOT / 'code/Problem1').glob('*.py'))
    source_paths += sorted((ROOT / 'code/tools').glob('*.py'))
    manifest = dict(
        python=sys.version, platform=platform.platform(),
        config_sha256=digest(config),
        input_sha256={path.stem: digest(path) for path in paths},
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in source_paths},
        reference_sha256={path.name: digest(path) for path in sorted(REFERENCE.glob('*.py'))},
        cases=len(paths), cores=CORES, workers=args.workers)
    write_json(args.output / 'manifest.json', manifest)

    started = time.perf_counter()
    rows, case_times, errors = [], [], []
    jobs = [(str(path), str(config), str(args.output)) for path in paths]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_case, job): Path(job[0]).stem for job in jobs}
        for future in as_completed(futures):
            case = futures[future]
            try:
                values, timing = future.result()
                rows.extend(values)
                case_times.append(timing)
                print(f'COMPLETE {case} ({timing["seconds"]:.1f}s)', flush=True)
            except Exception as error:
                errors.append(dict(case=case, error=repr(error)))
                print(f'FAILED {case}: {error!r}', flush=True)

    rows.sort(key=lambda row: (row['case'], row['cores']))
    metrics_path = args.output / 'metrics.csv'
    write_csv(metrics_path, rows)
    expected = {(f'case_{i:03}', cores) for i in range(1, 101) for cores in CORES}
    row_keys = {(row['case'], row['cores']) for row in rows}
    if len(rows) != 400 or row_keys != expected:
        errors.append(dict(case='summary', error='配置覆盖不完整或重复'))
    csv_rows = []
    if rows:
        with metrics_path.open('r', encoding='utf-8-sig', newline='') as stream:
            csv_rows = list(csv.DictReader(stream))
    csv_keys = {(row['case'], int(row['cores']), row['plan_sha256']) for row in csv_rows}
    if len(csv_rows) != len(rows) or csv_keys != {
            (row['case'], row['cores'], row['plan_sha256']) for row in rows}:
        errors.append(dict(case='summary', error='CSV 行数或方案哈希与内存记录不一致'))
    for row in rows:
        plan_path = args.output / row['case'] / f'core_{row["cores"]}' / f'{row["case"]}_multicore_res.json'
        if not plan_path.is_file() or digest(plan_path) != row['plan_sha256']:
            errors.append(dict(case=row['case'], error=f'core_{row["cores"]} 方案文件哈希不一致'))
    summary = dict(
        completed=len(case_times) == 100 and not errors,
        completed_cases=len(case_times), rows=len(rows), errors=errors,
        metrics_sha256=digest(metrics_path) if rows else None,
        wall_seconds=time.perf_counter() - started, case_times=case_times,
        mean_speedup={cores: statistics.mean(
            row['speedup'] for row in rows if row['cores'] == cores)
            for cores in CORES if any(row['cores'] == cores for row in rows)})
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

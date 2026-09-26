"""场景 B 的 100 个用例、2～5 核完整批量求解与官方核验。"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import platform
import statistics
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.experiment import digest, official_summary, write_csv
from tools.graph import Graph
from tools.official import REFERENCE, ROOT, SceneBSettings, read_json, write_json
from Problem2.solve import Solver


def run_case(job):
    path, config, output = job
    settings = SceneBSettings.read(config)
    solver = Solver(read_json(path), settings)
    singlecore = solver.single['makespan']
    family = solver.graph.features()['family']
    rows = []
    previous_score = None

    for cores in range(2, 6):
        begun = time.perf_counter()
        best, report = solver.solve(cores)
        if previous_score is not None and best.score > previous_score:
            raise AssertionError(f'{path.stem}: 增加核心后结果退化')
        previous_score = best.score
        directory = output / path.stem / f'core_{cores}'
        submission = directory / f'{path.stem}_multicore_res.json'
        write_json(submission, best.plan.as_dict())
        saved = read_json(submission)
        if saved != best.plan.as_dict():
            raise AssertionError(f'{submission}: 提交文件与求解方案不同')
        verification_started = time.perf_counter()
        verified = settings.evaluate(solver.graph.raw, saved)
        if verified != best.result:
            raise AssertionError(f'{submission}: 文件复评与求解结果不同')
        submission_verification = time.perf_counter() - verification_started
        write_json(directory / 'official.json', official_summary(verified))
        write_json(directory / 'search.json', report)

        movement = verified['data_movement_bytes']
        rows.append(dict(case=path.stem, family=family, cores=cores,
                         plan_sha256=digest(submission),
                         singlecore_makespan=singlecore, makespan=verified['makespan'],
                         speedup=singlecore / verified['makespan'],
                         added_bytes=movement['added_copy_bytes'],
                         partition_added_bytes=movement['partition_added_copy_bytes'],
                         spill_added_bytes=movement['spill_added_copy_bytes'],
                         scheduled_bytes=movement['scheduled_copy_bytes'],
                         subgraphs=sum(map(len, best.plan.blocks)),
                         active_cores=sum(bool(core) for core in best.plan.blocks),
                         search_seconds=report['search_seconds'],
                         verification_seconds=report['verification_seconds'] + submission_verification,
                         solve_seconds=time.perf_counter() - begun,
                         preview_calls=report['preview_calls'],
                         global_lower_bound=report['global_lower_bound'],
                         gap_upper_bound=report['gap_upper_bound'],
                         selected=best.origin, verified=True))
        print(f'{path.stem} N={cores}: T={best.score[0]}; '
              f'speedup={rows[-1]["speedup"]:.3f}', flush=True)
    return rows


def summarize(rows):
    result = []
    for cores in range(2, 6):
        values = [row for row in rows if row['cores'] == cores]
        result.append(dict(cores=cores, cases=len(values),
                           mean_speedup=statistics.mean(row['speedup'] for row in values),
                           median_speedup=statistics.median(row['speedup'] for row in values)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / '题目/data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error('workers 必须为正整数')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('输出目录必须不存在或为空')

    config = args.data / 'config.txt'
    cases = [args.data / f'case_{i:03d}.json' for i in range(1, 101)]
    missing = [path.name for path in [config, *cases] if not path.is_file()]
    if missing:
        parser.error(f'数据目录缺少 {len(missing)} 个必需文件：{", ".join(missing[:5])}')

    started = time.perf_counter()
    features = {path.stem: Graph(read_json(path)).features() for path in cases}
    sources = list((ROOT / 'code/tools').glob('*.py')) + list(Path(__file__).parent.glob('*.py'))
    write_json(args.output / 'manifest.json', dict(
        python=sys.version, platform=platform.platform(), workers=args.workers,
        config_sha256=digest(config),
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in sources},
        reference_sha256={path.name: digest(path) for path in REFERENCE.glob('*.py')},
        inputs={path.stem: digest(path) for path in cases},
        families=sorted({item['family'] for item in features.values()})))

    rows, failures = [], {}
    jobs = [(path, config, args.output)
            for path in sorted(cases, key=lambda item: -features[item.stem]['compute_ops'])]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, job): job[0].stem for job in jobs}
        for future in as_completed(futures):
            case = futures[future]
            try:
                rows.extend(future.result())
            except Exception as error:
                failures[case] = f'{type(error).__name__}: {error}'
                print(f'{case}: FAILED: {failures[case]}', flush=True)

    rows.sort(key=lambda row: (row['case'], row['cores']))
    write_csv(args.output / 'metrics.csv', rows)
    completed = not failures and len(rows) == 400
    summary = dict(completed=completed, rows=len(rows), requested_cases=100,
                   completed_cases=len({row['case'] for row in rows}),
                   wall_seconds=time.perf_counter() - started, failures=failures,
                   cores=summarize(rows) if completed else [])
    write_json(args.output / 'summary.json', summary)
    print(summary, flush=True)
    if not completed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

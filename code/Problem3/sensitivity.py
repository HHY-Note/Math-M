"""补充实验：固定本题已核验方案，内存中改变 L2 参数；不改正式配置和成绩。"""

from dataclasses import replace
from pathlib import Path
import argparse
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.official import CacheSettings, ROOT, REFERENCE, read_json, write_json
from tools.experiment import digest, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    parser.add_argument('--data', type=Path, default=ROOT / '题目/data')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--cases', type=int, nargs='+', required=True)
    parser.add_argument('--cores', type=int, default=5, choices=range(1, 6))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(set(args.cases)) != len(args.cases):
        parser.error('输出文件不得已存在；用例编号不得重复')
    config = args.config or args.data / 'config.txt'
    settings = CacheSettings.read(config)
    original = read_json(args.experiment / 'manifest.json')
    if digest(config) != original['config_sha256']:
        raise ValueError('输入配置与原实验不同')
    capacities = (settings.cache_capacity // 2, settings.cache_capacity, settings.cache_capacity * 2)
    bandwidths = (settings.cache_bandwidth / 2, settings.cache_bandwidth, settings.cache_bandwidth * 2)
    rows, inputs, plans = [], {}, {}
    started = time.perf_counter()
    for index in args.cases:
        case = f'case_{index:03d}'
        path = args.data / f'{case}.json'
        folder = args.experiment / case / f'core_{args.cores}'
        plan_path = folder / f'{case}_multicore_res.json'
        verified = read_json(folder / 'official.json')
        inputs[case], plans[case] = digest(path), digest(plan_path)
        if inputs[case] != original['inputs'][case] or plans[case] != verified['plan_sha256']:
            raise ValueError(f'{case}: 原图或已核验方案发生变化')
        graph, plan = read_json(path), read_json(plan_path)
        for capacity in capacities:
            for bandwidth in bandwidths:
                cfg = replace(settings, cache_capacity=capacity, cache_bandwidth=bandwidth)
                result = cfg.evaluate(graph, plan)
                if capacity == settings.cache_capacity and bandwidth == settings.cache_bandwidth:
                    if result['makespan'] != verified['l2']['makespan'] or result['cache_stats'] != verified['l2']['cache_stats']:
                        raise AssertionError('固定参数基准复算不一致')
                rows.append(dict(case=case, cores=args.cores, cache_capacity_bytes=capacity,
                                 cache_bandwidth=bandwidth, makespan=result['makespan'],
                                 hit_rate=result['cache_stats']['hit_rate'], hits=result['cache_stats']['hits'],
                                 added_bytes=result['data_movement_bytes']['added_copy_bytes'],
                                 official=True, supplementary=True))
                print(f'{case}: C={capacity}, B={bandwidth:g}, T={result["makespan"]}', flush=True)
        write_csv(args.output, rows)
    if digest(config) != original['config_sha256']:
        raise AssertionError('正式配置被修改')
    write_json(args.output.with_suffix('.manifest.json'),
               dict(supplementary=True, fixed_plan=True, config_sha256=digest(config), inputs=inputs,
                    plan_sha256=plans, capacities=capacities, bandwidths=bandwidths,
                    script_sha256=digest(__file__),
                    reference_sha256={p.name: digest(p) for p in REFERENCE.glob('*.py')},
                    wall_seconds=time.perf_counter() - started, rows=len(rows)))


if __name__ == '__main__':
    main()

"""只分析真实官方时间线；命中率按字节计算，不用其替代完工时间。"""

from collections import Counter, defaultdict
import statistics


def analyze(result, settings):
    inserted, requested = set(), set()
    misses, cold_duplicates, capacity_misses, bypass = Counter(), 0, 0, 0
    for event in result['cache_events']:
        key = event['tensor_id']
        if event['event'] == 'insert':
            inserted.add(key)
        elif event['event'] == 'miss':
            misses[key] += event['size_bytes']
            if event['size_bytes'] > settings.cache_capacity:
                bypass += 1
            elif key in inserted:
                capacity_misses += 1
            elif key in requested:
                cold_duplicates += 1
            requested.add(key)
    finish, busy = [], []
    for core in result['per_core_timeline']:
        finish.append(max((op['end'] for op in core['ops']), default=0))
        work = defaultdict(int)
        for op in core['ops']:
            if op['op'] not in ('COPY_IN', 'COPY_OUT'):
                work[op['pipe']] += op['duration']
        busy.append(dict(work))
    waits = [x['copy_in_start'] - x['copy_in_release'] for x in result['cross_core_transfers']]
    total = max(1, result['makespan'])
    ddr_bytes = result['data_movement_bytes']['scheduled_copy_bytes'] - result['cache_stats']['hit_bytes']
    return dict(core_finish=finish, compute_pipe_busy=busy, ddr_path_bytes=ddr_bytes,
                ddr_byte_demand_fraction=ddr_bytes / settings.bandwidth / total,
                cache_byte_demand_fraction=result['cache_stats']['hit_bytes'] / settings.cache_bandwidth / total,
                max_compute_pipe_fraction=max((max(x.values(), default=0) for x in busy), default=0) / total,
                cold_duplicate_misses=cold_duplicates, post_fill_misses=capacity_misses,
                oversized_misses=bypass, cross_transfers=len(waits),
                mean_copy_queue_wait=statistics.mean(waits) if waits else 0,
                max_copy_queue_wait=max(waits, default=0),
                costly_miss_tensors=[dict(tensor_id=t, miss_bytes=b) for t, b in misses.most_common(12)])


def compact(result):
    return {k: result[k] for k in ('makespan', 'num_cores', 'data_movement_bytes',
                                   'memory_peak_by_core', 'cache_stats') if k in result}

"""关键路径列表调度；通过少量合法迁移改善真实多核执行时间。"""

import heapq
import math

from Problem1.execution import check_schedules


def lower_bound(partition, cores):
    """固定划分的下界：Task 路径、总工作量及共享 DDR 的最少服务时间。"""
    finish = {}
    for s in partition.order:
        finish[s] = partition.duration[s] + max(
            (finish[p] for p in partition.preds[s]), default=0)
    return math.ceil(max(max(finish.values(), default=0),
                         sum(partition.duration) / cores, partition.ddr_work))


def list_schedule(partition, cores, settings, durations=None, priority='critical'):
    durations = durations if durations is not None else partition.duration
    rank = {}
    for s in reversed(partition.order):
        rank[s] = durations[s] + max((rank[w] for w in partition.succs[s]), default=0)
    degree = {s: len(p) for s, p in partition.preds.items()}
    ready = [(-(rank[s] if priority == 'critical' else durations[s]), s)
             for s, d in degree.items() if not d]
    heapq.heapify(ready)
    orders = [[] for _ in range(cores)]
    available, owner, finish = [0] * cores, {}, {}
    while ready:
        _, s = heapq.heappop(ready)
        choices = []
        for c in range(cores):
            local = available[c] + (settings.same_wait if orders[c] else 0)
            dependency = max((finish[r] + (settings.cross_wait if owner[r] != c else 0)
                              for r in partition.preds[s]), default=0)
            end = max(local, dependency) + durations[s]
            choices.append((end, available[c], c))
        end, _, c = min(choices)
        orders[c].append(s)
        available[c] = finish[s] = end
        owner[s] = c
        for w in sorted(partition.succs[s]):
            degree[w] -= 1
            if not degree[w]:
                heapq.heappush(ready, (-(rank[w] if priority == 'critical' else durations[w]), w))
    return orders, max(max(finish.values(), default=0), partition.ddr_work)


def neighbors(partition, schedules, result, limit=12):
    """优先移动晚完成的大任务；只产生通过全局 Task 判环的候选。"""
    if limit <= 0 or len(partition.groups) < 2:
        return
    owner = result['owner']
    tasks = sorted(partition.preds, key=lambda s: (
        -result['task_end'][s], -(result['task_end'][s] - result['task_start'][s]), s))[:4]
    seen, emitted = set(), 0
    for s in tasks:
        source = owner[s]
        for target in range(len(schedules)):
            base = [list(order) for order in schedules]
            base[source].remove(s)
            destination = base[target]
            near = sum(result['task_start'][w] <= result['task_start'][s] for w in destination)
            for position in sorted({0, near, len(destination)}):
                trial = [list(order) for order in base]
                trial[target].insert(position, s)
                key = tuple(map(tuple, trial))
                if key in seen or trial == schedules:
                    continue
                seen.add(key)
                try:
                    check_schedules(partition, trial)
                except ValueError:
                    continue
                yield trial
                emitted += 1
                if emitted >= limit:
                    return

"""缓存原 Step1/2/3 的输出；在固定执行图上模拟场景 A。

资源和取整规则与原评测器一致。优化只在实现开销：任务完成用计数器，
不在每个事件重新扫描任务内所有操作；运行态最多维护 4N 个操作。
"""

from collections import OrderedDict
from dataclasses import dataclass
import math

from tools.graph import topological
from tools.official import PIPES, _build_scene_a_tasks, _op_duration, _uses_ddr_bandwidth


def check_schedules(partition, schedules):
    flat = [s for order in schedules for s in order]
    if sorted(flat) != list(range(len(partition.groups))):
        raise ValueError('每个子图必须且只能安排一次')
    preds = {s: set(p) for s, p in partition.preds.items()}
    succs = {s: set(p) for s, p in partition.succs.items()}
    for order in schedules:
        for a, b in zip(order, order[1:]):
            preds[b].add(a)
            succs[a].add(b)
    topological(preds, succs)


@dataclass
class Compiled:
    partition: object
    durations: list
    op_ids: list
    task_of: list
    work: list
    ddr: list
    degree: list
    successors: list
    pipes: list
    task_sizes: list
    peaks: list
    movement: dict


class Engine:
    def __init__(self, graph, settings, cache_size=8):
        self.graph, self.settings = graph, settings
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.preparations = 0

    def prepare(self, partition):
        key = partition.groups
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        tasks, _, movement, _ = _build_scene_a_tasks(
            self.graph.raw, partition.plan([partition.order]),
            self.settings.bandwidth, self.settings.capacity)
        op_ids, task_of, work, ddr, degree, successors = [], [], [], [], [], []
        pipes, sizes, peaks, durations = [], [], [], []
        for s in range(len(partition.groups)):
            task = tasks.pop(s)
            seq = task['seq']
            index = {v: len(op_ids) + i for i, v in enumerate(seq)}
            for v in seq:
                op = task['op_by_id'][v]
                op_ids.append(v)
                task_of.append(s)
                work.append(_op_duration(op, task['in_tids'], task['out_tids'],
                                         task['tensor_by_id'], self.settings.bandwidth))
                ddr.append(_uses_ddr_bandwidth(op, task['in_tids'], task['out_tids'],
                                               task['tensor_by_id']))
                degree.append(len(task['op_preds'][v]))
                successors.append(tuple(index[w] for w in sorted(task['op_succs'][v])))
            pipes.append(tuple(tuple(index[v] for v in task['pipe_ops'][p]) for p in PIPES))
            sizes.append(len(seq))
            peaks.append(task['step3']['memory_peak'])
            durations.append(task['step3']['makespan'])
        result = Compiled(partition, durations, op_ids, task_of, work, ddr, degree,
                          successors, pipes, sizes, peaks, movement)
        self.preparations += 1
        self.cache[key] = result
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result

    def simulate(self, compiled, schedules):
        part, cfg = compiled.partition, self.settings
        check_schedules(part, schedules)
        n, m = len(schedules), len(part.groups)
        owner = [0] * m
        for c, order in enumerate(schedules):
            for s in order:
                owner[s] = c
        task_start, task_end = [-1] * m, [-1] * m
        remaining = compiled.task_sizes.copy()
        task_degree = [len(part.preds[s]) for s in range(m)]
        cross_release = [0] * m
        position, active = [0] * n, [-1] * n
        previous = [None] * n
        cursor = [[0] * 4 for _ in range(n)]
        running = [[-1] * 4 for _ in range(n)]
        degree = compiled.degree.copy()
        starts, ends = [-1] * len(degree), [-1] * len(degree)
        busy = [[0] * 4 for _ in range(n)]
        pool, last_update = {}, 0

        def settle(now):
            nonlocal last_update
            elapsed = now - last_update
            while elapsed > 1e-9:
                ids = [v for v, w in pool.items() if w > 1e-9]
                if not ids:
                    break
                minimum = min(pool[v] for v in ids)
                delta = minimum * len(ids)
                if delta >= elapsed - 1e-9:
                    share = elapsed / len(ids)
                    for v in ids:
                        pool[v] = max(0.0, pool[v] - share)
                    break
                for v in ids:
                    pool[v] = max(0.0, pool[v] - minimum)
                elapsed -= delta
            last_update = now

        def project(now):
            ordered = sorted((max(0.0, w), v) for v, w in pool.items())
            elapsed, previous_work, count, i = float(now), 0.0, len(ordered), 0
            while i < len(ordered):
                w = ordered[i][0]
                elapsed += (w - previous_work) * count
                j = i
                while j < len(ordered) and abs(ordered[j][0] - w) <= 1e-9:
                    ends[ordered[j][1]] = int(math.ceil(elapsed - 1e-9))
                    j += 1
                count -= j - i
                previous_work, i = w, j

        def release(c, s):
            if task_degree[s]:
                return None
            local = 0 if previous[c] is None else previous[c] + cfg.same_wait
            return max(local, cross_release[s])

        now, completed = 0, 0
        while completed < m:
            settle(now)
            retired_ddr = False
            for c in range(n):
                for p in range(4):
                    v = running[c][p]
                    if v < 0 or ends[v] > now:
                        continue
                    running[c][p] = -1
                    cursor[c][p] += 1
                    busy[c][p] += ends[v] - starts[v]
                    remaining[compiled.task_of[v]] -= 1
                    if v in pool:
                        del pool[v]
                        retired_ddr = True
                    for w in compiled.successors[v]:
                        degree[w] -= 1
            if retired_ddr:
                project(now)
            for c in range(n):
                s = active[c]
                if s >= 0 and remaining[s] == 0:
                    task_end[s] = now
                    previous[c] = now
                    active[c] = -1
                    position[c] += 1
                    completed += 1
                    for w in part.succs[s]:
                        task_degree[w] -= 1
                        if owner[w] != c:
                            cross_release[w] = max(cross_release[w], now + cfg.cross_wait)
            for c in range(n):
                if active[c] >= 0 or position[c] == len(schedules[c]):
                    continue
                s = schedules[c][position[c]]
                ready_at = release(c, s)
                if ready_at is not None and ready_at <= now:
                    active[c] = s
                    task_start[s] = now
                    cursor[c] = [0] * 4
            for c in range(n):
                s = active[c]
                if s < 0:
                    continue
                for p, order in enumerate(compiled.pipes[s]):
                    if running[c][p] >= 0 or cursor[c][p] == len(order):
                        continue
                    v = order[cursor[c][p]]
                    if degree[v]:
                        continue
                    running[c][p] = v
                    starts[v], ends[v] = now, now + compiled.work[v]
                    if compiled.ddr[v]:
                        settle(now)
                        pool[v] = float(compiled.work[v])
                        project(now)
            if completed == m:
                break
            next_times = [ends[v] for row in running for v in row if v >= 0]
            for c in range(n):
                if active[c] < 0 and position[c] < len(schedules[c]):
                    ready_at = release(c, schedules[c][position[c]])
                    if ready_at is not None and ready_at > now:
                        next_times.append(ready_at)
            if not next_times or min(next_times) <= now:
                raise RuntimeError('执行图未能推进；检查依赖、Pipe 顺序与任务顺序')
            now = min(next_times)
        return dict(makespan=now, task_start=task_start, task_end=task_end,
                    op_start=starts, op_end=ends, owner=owner, pipe_busy=busy,
                    data_movement_bytes=compiled.movement,
                    ddr_full_bandwidth_work=sum(w for w, d in zip(compiled.work, compiled.ddr) if d),
                    memory_peak_by_core={c: {
                        pos: max((compiled.peaks[s][pos] for s in schedules[c]), default=0)
                        for pos in cfg.capacity} for c in range(n)})


def assert_official(compiled, fast, official):
    """最终验收比较每个操作时间，而不只比较 Makespan。"""
    for key in ('makespan', 'data_movement_bytes', 'memory_peak_by_core'):
        if fast[key] != official[key]:
            raise AssertionError(f'与原评测器不一致: {key}')
    index = {(s, v): i for i, (s, v) in enumerate(zip(compiled.task_of, compiled.op_ids))}
    count = 0
    for core in official['per_core_timeline']:
        for task in core['tasks']:
            s = task['task_id']
            if (task['start'], task['end']) != (fast['task_start'][s], fast['task_end'][s]):
                raise AssertionError(f'Task 时间不一致: {s}')
        for op in core['ops']:
            i = index[op['task_id'], op['op_id']]
            if (op['start'], op['end']) != (fast['op_start'][i], fast['op_end'][i]):
                raise AssertionError(f'操作时间不一致: {op}')
            count += 1
    if count != len(index):
        raise AssertionError('评测操作覆盖数不一致')

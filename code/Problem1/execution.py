"""缓存原 Step1/2/3 的输出；在固定执行图上模拟场景 A。

资源和取整规则与原评测器一致。优化只在实现开销：任务完成用计数器，
不在每个事件重新扫描任务内所有操作；运行态最多维护 4N 个操作。
"""

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import math

from tools.graph import topological
from tools.official import (
    PIPES, _op_duration, _uses_ddr_bandwidth, _copy_traffic_bytes,
    step1_schedule, step2_spill_insertion, _build_extended_graph, prepare_step3_execution,
)


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


@dataclass
class Kernel:
    """固定节点集合的核内执行图；边界 COPY 与 SPILL 的编号可按序平移。"""
    op_ids: list
    copy_ids: list
    spill_base: int
    work: list
    ddr: list
    degree: list
    successors: list
    pipes: tuple
    peak: dict
    duration: int
    spill_bytes: int


class Engine:
    def __init__(self, graph, settings, cache_size=8):
        self.graph, self.settings = graph, settings
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.preparations = 0
        self.kernels = OrderedDict()
        self.kernel_ops = 0
        self.kernel_limit = 8 * max(32, len(graph.all_ops))
        self.kernel_builds = self.kernel_hits = 0
        self.original_bytes = _copy_traffic_bytes(graph.raw)
        self.direct = defaultdict(list)
        for i, edge in enumerate(graph.raw['edges']):
            if edge['source'] in graph.ops and edge['target'] in graph.ops:
                self.direct[edge['source']].append((i, edge))

    def _kernel(self, nodes, tensors, boundaries):
        """只对新子图调用原 Step1/2/3；缓存与核指派及其他子图的划分无关。"""
        if nodes in self.kernels:
            self.kernel_hits += 1
            self.kernels.move_to_end(nodes)
            return self.kernels[nodes]
        graph, cfg = self.graph, self.settings
        members = set(nodes)
        ops = [dict(graph.ops[v]) for v in nodes]
        local_tensors, edges = [], []
        for t in tensors:
            tensor = dict(graph.tensors[t])
            if tensor['pos'] == 'DDR':
                tensor['pos'] = 'UB'
            local_tensors.append(tensor)
            edges.extend(dict(source=v, target=t) for v in sorted(graph.producers[t] & members))
            edges.extend(dict(source=t, target=v) for v in sorted(graph.consumers[t] & members))
            for tid, copy_id, incoming in boundaries.get(t, ()):
                size = tensor['size']
                local_tensors.append(dict(id=tid, pos='DDR', size=size))
                ops.append(dict(id=copy_id, op='COPY_IN' if incoming else 'COPY_OUT',
                                pipe=PIPES[0 if incoming else 1],
                                cycles=max(1, math.ceil(size / cfg.bandwidth))))
                source, target = (tid, t) if incoming else (t, tid)
                edges.extend((dict(source=source, target=copy_id),
                              dict(source=copy_id, target=target)))
        edges.extend(dict(edge) for _, edge in sorted(
            (i, e) for v in nodes for i, e in self.direct[v] if e['target'] in members))
        local = dict(ops=ops, tensors=local_tensors, edges=edges)
        spill_base = max([o['id'] for o in ops] + [t['id'] for t in local_tensors] + [0]) + 1
        spills = step2_spill_insertion(local, step1_schedule(local), capacity=cfg.capacity)
        task = prepare_step3_execution(_build_extended_graph(local, spills),
                                       capacity=cfg.capacity, bandwidth=cfg.bandwidth)
        seq = task['seq']
        index = {v: i for i, v in enumerate(seq)}
        work, ddr = [], []
        for v in seq:
            args = (task['op_by_id'][v], task['in_tids'], task['out_tids'], task['tensor_by_id'])
            work.append(_op_duration(*args, cfg.bandwidth))
            ddr.append(_uses_ddr_bandwidth(*args))
        kernel = Kernel(
            seq, [cid for row in boundaries.values() for _, cid, _ in row], spill_base,
            work, ddr, [len(task['op_preds'][v]) for v in seq],
            [tuple(index[w] for w in sorted(task['op_succs'][v])) for v in seq],
            tuple(tuple(index[v] for v in task['pipe_ops'][p]) for p in PIPES),
            task['step3']['memory_peak'], task['step3']['makespan'],
            sum(s['size'] * (1 + int(s['spill_out_copies_data'])) for s in spills['spill_records']))
        self.kernel_builds += 1
        self.kernels[nodes] = kernel
        self.kernel_ops += len(seq)
        while self.kernel_ops > self.kernel_limit and len(self.kernels) > 1:
            self.kernel_ops -= len(self.kernels.popitem(last=False)[1].op_ids)
        return kernel

    def prepare(self, partition):
        key = partition.groups
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        graph = self.graph
        used = set(graph.all_ops) | set(graph.tensors)
        next_op = max(graph.all_ops, default=0) + 1
        next_tensor = max([10000, *graph.tensors]) + 1
        op_ids, task_of, work, ddr, degree, successors = [], [], [], [], [], []
        pipes, sizes, peaks, durations = [], [], [], []
        boundary_bytes = spill_bytes = 0
        for s, nodes in enumerate(partition.groups):
            members = set(nodes)
            touched = sorted({t for v in nodes for t in graph.inputs[v] | graph.outputs[v]})
            boundaries = {}
            for t in touched:
                ps, cs = graph.producers[t] & members, graph.consumers[t] & members
                directions = []
                if cs and not ps:
                    directions.append(True)
                if ps and (t in graph.final or not graph.consumers[t] or graph.consumers[t] - members):
                    directions.append(False)
                for incoming in directions:
                    while next_tensor in used:
                        next_tensor += 1
                    tid = next_tensor
                    used.add(tid)
                    next_tensor += 1
                    while next_op in used:
                        next_op += 1
                    copy_id = next_op
                    used.add(copy_id)
                    next_op += 1
                    boundaries.setdefault(t, []).append((tid, copy_id, incoming))
                    boundary_bytes += graph.tensors[t]['size']
            kernel = self._kernel(nodes, touched, boundaries)
            copies = [cid for row in boundaries.values() for _, cid, _ in row]
            rename = dict(zip(kernel.copy_ids, copies))
            spill_base = max([*nodes, *copies, *touched,
                              *(tid for row in boundaries.values() for tid, _, _ in row), 0]) + 1
            offset = len(op_ids)
            op_ids.extend(rename.get(v, v + spill_base - kernel.spill_base
                                     if v >= kernel.spill_base else v) for v in kernel.op_ids)
            task_of.extend([s] * len(kernel.op_ids))
            work.extend(kernel.work)
            ddr.extend(kernel.ddr)
            degree.extend(kernel.degree)
            successors.extend(tuple(offset + w for w in row) for row in kernel.successors)
            pipes.append(tuple(tuple(offset + v for v in row) for row in kernel.pipes))
            sizes.append(len(kernel.op_ids))
            peaks.append(kernel.peak)
            durations.append(kernel.duration)
            spill_bytes += kernel.spill_bytes
        movement = dict(original_graph_copy_bytes=self.original_bytes,
                        scheduled_copy_bytes=boundary_bytes + spill_bytes,
                        added_copy_bytes=boundary_bytes + spill_bytes - self.original_bytes,
                        partition_added_copy_bytes=boundary_bytes - self.original_bytes,
                        spill_added_copy_bytes=spill_bytes)
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

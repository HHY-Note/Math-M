"""少量可解释的初始划分，以及按张量去重的快速成本。"""

from dataclasses import dataclass
from collections import defaultdict
from bisect import bisect_left
import math

from tools.graph import topological
from tools.official import PIPES, step1_schedule


def canonical(groups):
    return tuple(sorted((tuple(sorted(g)) for g in groups if g), key=lambda g: g[0]))


@dataclass
class Partition:
    groups: tuple
    name: str
    owner: dict
    preds: dict
    succs: dict
    order: list
    duration: list
    copy_bytes: int
    ddr_work: int

    def plan(self, schedules):
        return {'node_to_subgraph': {str(v): s for v, s in sorted(self.owner.items())},
                'core_schedules': [list(order) for order in schedules]}


def describe(graph, groups, settings, name):
    groups = canonical(groups)
    owner = {v: s for s, nodes in enumerate(groups) for v in nodes}
    if len(owner) != sum(map(len, groups)) or set(owner) != set(graph.ops):
        raise ValueError('划分必须恰好覆盖全部非 COPY 操作')
    preds = {s: set() for s in range(len(groups))}
    succs = {s: set() for s in preds}
    for v, following in graph.succs.items():
        for w in following:
            a, b = owner[v], owner[w]
            if a != b:
                preds[b].add(a)
                succs[a].add(b)
    order = topological(preds, succs)
    load = [[0] * 4 for _ in groups]
    ddr_work = [0] * len(groups)
    for v, op in graph.ops.items():
        load[owner[v]][PIPES.index(op['pipe'])] += max(1, op['cycles'])
    copy_bytes = 0
    read_time, write_time = {}, {}
    for tid, tensor in graph.tensors.items():
        ps = {owner[v] for v in graph.producers[tid]}
        cs = {owner[v] for v in graph.consumers[tid]}
        size = tensor['size']
        duration = max(1, math.ceil(size / settings.bandwidth))
        for s in cs - ps:
            load[s][0] += duration
            ddr_work[s] += duration
            read_time[s, tid] = duration
            copy_bytes += size
        for s in ps:
            if tid in graph.final or not cs or cs - {s}:
                load[s][1] += duration
                ddr_work[s] += duration
                write_time[s, tid] = duration
                copy_bytes += size
    finish, critical = {}, [0] * len(groups)
    for v in graph.order:
        s = owner[v]
        start = max((finish[u] for u in graph.preds[v] if owner[u] == s), default=0)
        start = max(start, max((read_time.get((s, t), 0)
                               for t in graph.inputs[v]), default=0))
        finish[v] = start + max(1, graph.ops[v]['cycles'])
        end = finish[v] + max((write_time.get((s, t), 0)
                              for t in graph.outputs[v]), default=0)
        critical[s] = max(critical[s], end)
    duration = [max(critical[s], max(load[s]), ddr_work[s])
                for s in range(len(groups))]
    return Partition(groups, name, owner, preds, succs, order, duration,
                     copy_bytes, sum(ddr_work))


def chunks(units, graph, count):
    """连续拓扑单元按计算周期切分；不在单元内部截断。"""
    if not units:
        return ()
    weights = [sum(graph.ops[v]['cycles'] for v in unit) for unit in units]
    target = sum(weights) / max(1, count)
    groups, current, used = [], [], 0
    for unit, weight in zip(units, weights):
        if current and used >= target and len(groups) < count - 1:
            groups.append(current)
            current, used = [], 0
        current.extend(unit)
        used += weight
    groups.append(current)
    return canonical(groups)


def pack_components(components, graph, count, bandwidth, affinity=0.0):
    """独立分支装箱；同一分支不切开，因此跨箱不会产生依赖环。"""
    if not components:
        return ()
    count = min(count, len(components))
    groups = [[] for _ in range(count)]
    loads = [[0] * 4 for _ in groups]
    seen_inputs = [set() for _ in groups]
    items = []
    for nodes in components:
        work = [0] * 4
        inputs = set()
        node_set = set(nodes)
        for v in nodes:
            work[PIPES.index(graph.ops[v]['pipe'])] += graph.ops[v]['cycles']
            inputs.update(t for t in graph.inputs[v]
                          if not graph.producers[t] & node_set)
        items.append((nodes, work, inputs))
    items.sort(key=lambda item: (-max(item[1]), item[0][0]))
    for nodes, work, inputs in items:
        def score(k):
            finish = max(a + b for a, b in zip(loads[k], work))
            fresh = sum(graph.tensors[t]['size'] for t in inputs - seen_inputs[k])
            return finish + affinity * fresh / bandwidth, sum(loads[k]), k
        k = min(range(count), key=score)
        groups[k].extend(nodes)
        loads[k] = [a + b for a, b in zip(loads[k], work)]
        seen_inputs[k].update(inputs)
    return canonical(groups)


def layers(graph, cores, settings, cuts, affinity=0.0):
    """阶段内保持独立分支完整；阶段之间只保留原图的真实依赖。"""
    bands = defaultdict(list)
    for v in graph.order:
        bands[bisect_left(cuts, graph.depth[v])].append(v)
    groups = []
    for nodes in bands.values():
        groups.extend(pack_components(graph.connected(nodes), graph, cores,
                                      settings.bandwidth, affinity))
    return canonical(groups)


def affinity_partition(graph, count, settings):
    """沿安全链的拓扑序放置；通信增量按张量及消费子图去重。"""
    groups = [[] for _ in range(count)]
    loads = [[0] * 4 for _ in groups]
    owner, producers, consumers = {}, defaultdict(set), defaultdict(set)

    def copies(t, ps, cs):
        return len(cs - ps) + sum(t in graph.final or not graph.consumers[t]
                                  or bool(cs - {s}) for s in ps)

    for chain in graph.chains():
        nodes = set(chain)
        parents = {p for v in chain for p in graph.preds[v] if p not in nodes}
        low = max((owner[p] for p in parents), default=0)
        ins = set().union(*(graph.inputs[v] for v in chain))
        outs = set().union(*(graph.outputs[v] for v in chain))
        work = [0] * 4
        for v in chain:
            work[PIPES.index(graph.ops[v]['pipe'])] += max(1, graph.ops[v]['cycles'])

        def score(s):
            peak = max(max(a + b for a, b in zip(loads[s], work)),
                       max((max(loads[k]) for k in range(count) if k != s), default=0))
            delta = 0
            for t in ins | outs:
                ps, cs = producers[t], consumers[t]
                new_ps = ps | {s} if t in outs else ps
                new_cs = cs | {s} if t in ins else cs
                delta += (copies(t, new_ps, new_cs) - copies(t, ps, cs)) * graph.tensors[t]['size']
            return peak + delta / settings.bandwidth, sum(loads[s]), s

        s = min(range(low, count), key=score)
        groups[s].extend(chain)
        loads[s] = [a + b for a, b in zip(loads[s], work)]
        owner.update((v, s) for v in chain)
        for t in ins:
            consumers[t].add(s)
        for t in outs:
            producers[t].add(s)
    return canonical(groups)


def refinements(graph, cores, settings):
    """子图数随结构变化；固定窗口、分支/归约边界和张量亲和互为补充。"""
    depth = max(graph.depth.values(), default=0)
    if depth >= 8:
        for width in (2, 4, 8, 16):
            if width < depth:
                cuts = list(range(width, depth, width))
                yield f'window_{width}', layers(graph, cores, settings, cuts)
        widths = defaultdict(int)
        for d in graph.depth.values():
            widths[d] += 1
        cuts = set()
        for v in graph.order:
            d = graph.depth[v]
            incoming = sum(graph.tensors[t]['size'] for t in graph.inputs[v])
            outgoing = sum(graph.tensors[t]['size'] for t in graph.outputs[v])
            expands = len(graph.succs[v]) > 1 and widths[d + 1] >= max(cores, 2 * widths[d])
            shrinks = incoming > settings.bandwidth and 4 * outgoing <= incoming
            if d < depth and (expands or shrinks):
                cuts.add(d)
        if cuts:
            yield 'structure', layers(graph, cores, settings, sorted(cuts), affinity=0.5)
    for scale in (1, 2):
        yield f'affinity_{scale}', affinity_partition(graph, cores * scale, settings)


def candidates(graph, cores, settings):
    """独立分支、完整链、细粒度、层段四种结构；不使用用例编号。"""
    yield 'whole', canonical([graph.order])
    if not graph.ops:
        return
    chains = graph.chains()
    chain_of = {v: i for i, chain in enumerate(chains) for v in chain}
    dfs = [v for v in step1_schedule(graph.raw) if v in graph.ops]
    chain_order = list(dict.fromkeys(chain_of[v] for v in dfs))
    units = [chains[i] for i in chain_order]
    for scale in (1, 2, 4):
        k = cores * scale
        yield f'chains_{scale}', chunks(units, graph, k)
        yield f'topo_{scale}', chunks([[v] for v in graph.order], graph, k)
        if len(graph.components) > 1:
            yield f'components_{scale}', pack_components(
                graph.components, graph, k, settings.bandwidth)
    if len(graph.components) > 1:
        yield 'components_affinity', pack_components(
            graph.components, graph, cores, settings.bandwidth, affinity=0.5)
    max_depth = max(graph.depth.values())
    for stages in (2, 4, 8):
        cuts = [math.ceil(max_depth * j / stages) for j in range(1, stages)]
        yield f'levels_{stages}', layers(graph, cores, settings, cuts)


def feedback(graph, partition, schedules, result):
    """根据实测任务时长拆分瓶颈，或合并同核相邻任务；合法性由 describe 检查。"""
    if not partition.groups:
        return
    task = max(range(len(partition.groups)),
               key=lambda s: result['task_end'][s] - result['task_start'][s])
    nodes = partition.groups[task]
    if len(nodes) > 1:
        components = graph.connected(nodes)
        if len(components) > 1:
            parts = pack_components(components, graph, 2, 1)
        else:
            allowed = set(nodes)
            parts = chunks([[v] for v in graph.order if v in allowed], graph, 2)
        if len(parts) == 2:
            yield 'split_bottleneck', canonical(
                [g for s, g in enumerate(partition.groups) if s != task] + list(parts))
    pairs = [(a, b) for order in schedules for a, b in zip(order, order[1:])]
    if pairs:
        a, b = min(pairs, key=lambda ab: sum(
            result['task_end'][s] - result['task_start'][s] for s in ab))
        yield 'merge_neighbors', canonical(
            [g for s, g in enumerate(partition.groups) if s not in (a, b)]
            + [partition.groups[a] + partition.groups[b]])


def bottleneck_bands(graph, partition, result, cores, settings):
    """仅拆分占主导的大 Task；按内部深度分段，再将每段的独立分支装箱。"""
    if not partition.groups:
        return
    task = max(partition.preds, key=lambda s: result['task_end'][s] - result['task_start'][s])
    if cores * (result['task_end'][task] - result['task_start'][task]) < result['makespan']:
        return
    nodes = partition.groups[task]
    depths = sorted({graph.depth[v] for v in nodes})
    cuts = depths[:-1]
    if len(cuts) > 32:
        cuts = [cuts[round(i * (len(cuts) - 1) / 31)] for i in range(32)]
    fixed = [g for s, g in enumerate(partition.groups) if s != task]
    for cut in cuts:
        groups = list(fixed)
        for before in (True, False):
            band = [v for v in nodes if (graph.depth[v] <= cut) == before]
            groups.extend(pack_components(graph.connected(band), graph, cores, settings.bandwidth))
        yield f'bottleneck_band_{cut}', canonical(groups)

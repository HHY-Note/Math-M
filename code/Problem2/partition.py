"""场景 B 的方案表示、按张量计费和结构初始解。"""

from collections import defaultdict
from dataclasses import dataclass
from bisect import bisect_left
import math

from tools.graph import topological
from tools.official import PIPES, derive_multicore_plan, step1_schedule


@dataclass(frozen=True)
class Plan:
    """blocks[core] 是该核心按执行优先级排列的子图；空核心保留。"""
    blocks: tuple

    @classmethod
    def build(cls, blocks):
        return cls(tuple(tuple(tuple(sorted(group)) for group in core if group)
                         for core in blocks))

    @classmethod
    def read(cls, graph, value):
        view = derive_multicore_plan(graph.raw, value)
        return cls.build([[view['nodes_by_subgraph'][s] for s in order]
                          for order in value['core_schedules']])

    def as_dict(self):
        mapping, schedules = {}, []
        for core in self.blocks:
            order = []
            for group in core:
                sg = sum(map(len, schedules)) + len(order)
                order.append(sg)
                mapping.update((str(v), sg) for v in group)
            schedules.append(order)
        return dict(node_to_subgraph=dict(sorted(mapping.items(), key=lambda x: int(x[0]))),
                    core_schedules=schedules)

    def assignment(self):
        return {v: c for c, groups in enumerate(self.blocks) for group in groups for v in group}

    def padded(self, cores):
        if cores < len(self.blocks):
            raise ValueError('不能通过补空核减少核心数')
        return Plan(self.blocks + ((),) * (cores - len(self.blocks)))


def block_graph(graph, plan):
    items = [(c, group) for c, core in enumerate(plan.blocks) for group in core]
    owner = {v: s for s, (_, group) in enumerate(items) for v in group}
    if len(owner) != sum(len(group) for _, group in items) or set(owner) != set(graph.ops):
        raise ValueError('方案必须恰好覆盖全部非 COPY 操作')
    preds, succs = {s: set() for s in range(len(items))}, {s: set() for s in range(len(items))}
    for v in graph.order:
        for w in graph.succs[v]:
            a, b = owner[v], owner[w]
            if a != b:
                preds[b].add(a)
                succs[a].add(b)
    return items, preds, succs


def check_plan(graph, plan):
    items, preds, succs = block_graph(graph, plan)
    topological(preds, succs)
    for a, after in succs.items():
        if any(items[a][0] == items[b][0] and a > b for b in after):
            raise ValueError('同核子图顺序违反依赖')


def arrange(graph, groups, cores, priority):
    """将同一个子图 DAG 的拓扑序投影到各核；不添加 Task 完成屏障。"""
    blocks = [[] for _ in range(cores)]
    for core, group in groups:
        if group:
            blocks[core].append(group)
    plan = Plan.build(blocks)
    items, preds, succs = block_graph(graph, plan)
    ranks = {s: min(priority[v] for v in group) for s, (_, group) in enumerate(items)}
    ordered = [[] for _ in range(cores)]
    for s in topological(preds, succs, ranks):
        c, group = items[s]
        ordered[c].append(group)
    return Plan.build(ordered)


def runs(graph, owner, cores, order, width=256, limit=8192):
    """全局拓扑序的同核连续段保证子图 DAG 无环；限制只控制搜索规模。"""
    groups, previous = [], None
    for v in order:
        core = owner[v]
        if core != previous or len(groups[-1][1]) >= width:
            if len(groups) >= limit:
                return None
            groups.append((core, []))
            previous = core
        groups[-1][1].append(v)
    blocks = [[] for _ in range(cores)]
    for core, group in groups:
        blocks[core].append(group)
    return Plan.build(blocks)


class CostModel:
    """按张量估计通信与时间；忽略 Spill、FIFO 和内存等待的松弛给出时间下界。"""
    def __init__(self, graph, settings):
        self.graph, self.settings = graph, settings
        self.position = {v: i for i, v in enumerate(graph.order)}
        self.pipe = {v: PIPES.index(op['pipe']) for v, op in graph.ops.items()}
        self.records, self.transfer = {}, defaultdict(dict)
        for t, tensor in graph.tensors.items():
            ps, cs = graph.producers[t], graph.consumers[t]
            if not ps and not cs:
                continue
            size = tensor['size']
            work = max(1, math.ceil(size / settings.bandwidth))
            self.records[t] = (ps, cs, size, work, t in graph.final or not cs)
            for v in ps:
                for w in cs:
                    self.transfer[w][v] = max(work, self.transfer[w].get(v, 0))
        self.direct = [(e['source'], e['target'], int(e.get('data_size', 0)))
                       for e in graph.raw['edges']
                       if e['source'] in graph.ops and e['target'] in graph.ops]
        self.critical = topological(graph.preds, graph.succs,
                                    {v: -graph.rank[v] for v in graph.ops})
        release = {}
        for v in graph.ops:
            benefit = sum(graph.tensors[t]['size'] / settings.capacity[graph.tensors[t]['pos']]
                          for t in graph.inputs[v]
                          if graph.tensors[t]['pos'] in settings.capacity
                          and settings.capacity[graph.tensors[t]['pos']] > 0
                          and len(graph.consumers[t]) == 1)
            allocation = sum(graph.tensors[t]['size'] / settings.capacity[graph.tensors[t]['pos']]
                             for t in graph.outputs[v]
                             if graph.tensors[t]['pos'] in settings.capacity
                             and settings.capacity[graph.tensors[t]['pos']] > 0)
            release[v] = (allocation - benefit, -graph.rank[v])
        self.release = topological(graph.preds, graph.succs, release)
        self.chains = graph.chains()
        self.dfs = [v for v in step1_schedule(graph.raw) if v in graph.ops]
        self.units, self.bands, self.bounds = {}, {}, {}
        represented = {v: set(self.transfer[v]) for v in graph.ops}
        for v, w, _ in self.direct:
            represented[w].add(v)
        self.path_bound_valid = all(graph.preds[v] <= represented[v] for v in graph.ops)
        self.total_work = [sum(max(1, graph.ops[v]['cycles']) for v in graph.ops if self.pipe[v] == p)
                           for p in range(4)]
        self.minimum_ddr = sum(work * (int(bool(cs) and not ps) + int(bool(ps) and final))
                               for ps, cs, _, work, final in self.records.values())

    def lower_bound(self, cores):
        """所有分核方案的下界：计算负载均分、至少一次的图输入输出、原图计算路径。"""
        path = max(self.graph.rank.values(), default=0) if self.path_bound_valid else 0
        return max(math.ceil(max(self.total_work) / cores), self.minimum_ddr, path)

    def estimate(self, plan):
        graph, cfg = self.graph, self.settings
        owner, n = plan.assignment(), len(plan.blocks)
        load, traffic, ddr = [[0] * 4 for _ in range(n)], 0, 0
        for v, c in owner.items():
            load[c][self.pipe[v]] += max(1, graph.ops[v]['cycles'])
        for ps, cs, size, work, final in self.records.values():
            sources, targets = {owner[v] for v in ps}, {owner[v] for v in cs}
            reads, writes = [], []
            if not ps:
                reads.extend(targets)
            if ps and final:
                writes.extend(sources)
            for p in sources:
                for c in targets:
                    if p != c:
                        writes.append(p)
                        reads.append(c)
            for c in reads:
                load[c][0] += work
            for c in writes:
                load[c][1] += work
            traffic += size * (len(reads) + len(writes))
            ddr += work * (len(reads) + len(writes))
        for v, w, size in self.direct:
            if owner[v] != owner[w]:
                work = max(1, math.ceil(size / cfg.bandwidth))
                load[owner[v]][1] += work
                load[owner[w]][0] += work
                ddr += 2 * work
                traffic += 2 * size
        resource_bound = max(ddr, max(map(max, load), default=0))
        finish = {}
        for v in graph.order:
            start = max((finish[p] + (cfg.copy_delay + 2 * self.transfer[v].get(p, 1)
                                     if owner[p] != owner[v] else 0)
                         for p in graph.preds[v]), default=0)
            finish[v] = start + max(1, graph.ops[v]['cycles'])
        estimate = max(resource_bound, max(finish.values(), default=0))
        self.bounds[plan] = estimate if self.path_bound_valid else resource_bound
        return estimate, traffic

    def unit(self, nodes):
        """分组的 Pipe 工作量、外部输入和独占时间估计，供初始调度使用。"""
        nodes = tuple(sorted(nodes))
        if nodes not in self.units:
            graph, members = self.graph, set(nodes)
            work, inputs, outputs = [0] * 4, set(), set()
            for v in nodes:
                work[self.pipe[v]] += max(1, graph.ops[v]['cycles'])
                inputs.update(t for t in graph.inputs[v] if not graph.producers[t] & members)
                outputs.update(t for t in graph.outputs[v]
                               if t in graph.final or not graph.consumers[t]
                               or graph.consumers[t] - members)
            read = sum(self.records[t][3] for t in inputs)
            write = sum(self.records[t][3] for t in outputs)
            end, critical = {}, 0
            for v in sorted(nodes, key=self.position.get):
                start = max((end[p] for p in graph.preds[v] if p in members), default=0)
                start = max(start, max((self.records[t][3] for t in graph.inputs[v] & inputs), default=0))
                end[v] = start + max(1, graph.ops[v]['cycles'])
                critical = max(critical, end[v] + max((self.records[t][3]
                               for t in graph.outputs[v] & outputs), default=0))
            duration = max(critical, max(work), read + write, work[0] + read, work[1] + write)
            self.units[nodes] = work, inputs, duration
        return self.units[nodes]

    def pack(self, units, count, affinity=0.0):
        """对独立分支按最长工作量优先装箱；输入共享按张量去重。"""
        count = min(count, len(units))
        groups, loads, seen = [[] for _ in range(count)], [[0] * 4 for _ in range(count)], [set() for _ in range(count)]
        items = [(tuple(sorted(nodes)), self.unit(nodes)) for nodes in units]
        items.sort(key=lambda item: (-max(item[1][0]), item[0][0]))
        for nodes, (work, inputs, _) in items:
            def cost(c):
                fresh = sum(self.records[t][2] for t in inputs - seen[c])
                return (max(a + b for a, b in zip(loads[c], work))
                        + affinity * fresh / self.settings.bandwidth, sum(loads[c]), c)
            c = min(range(count), key=cost)
            groups[c].extend(nodes)
            loads[c] = [a + b for a, b in zip(loads[c], work)]
            seen[c].update(inputs)
        return [tuple(sorted(g)) for g in groups if g]

    def dispatch(self, groups, cores, affinity=1.0):
        """分组 DAG 的关键路径列表分核；估计显式计入跨核同步延迟。"""
        groups = sorted((tuple(sorted(g)) for g in groups if g), key=lambda g: g[0])
        _, preds, succs = block_graph(self.graph, Plan.build([groups]))
        order = topological(preds, succs)
        durations = [self.unit(g)[2] for g in groups]
        group_of = {v: s for s, g in enumerate(groups) for v in g}
        inbound = [self.unit(g)[1] for g in groups]
        rank = {}
        for s in reversed(order):
            rank[s] = durations[s] + max((rank[w] for w in succs[s]), default=0)
        available, finish, owner = [0] * cores, {}, {}
        blocks = [[] for _ in range(cores)]
        for s in topological(preds, succs, {s: -rank[s] for s in preds}):
            def choice(c):
                ready = max((finish[p] + (self.settings.copy_delay if owner[p] != c else 0)
                             for p in preds[s]), default=0)
                cross = sum(self.records[t][3] * len({owner[group_of[p]] for p in self.graph.producers[t]} - {c})
                            for t in inbound[s] if self.graph.producers[t])
                end = max(available[c], ready) + durations[s]
                return end + affinity * 2 * cross, end, available[c], c
            _, end, _, c = min(choice(c) for c in range(cores))
            finish[s] = available[c] = end
            owner[s] = c
            blocks[c].append(groups[s])
        return Plan.build(blocks)

    def layers(self, cores, cuts, affinity=0.0, resident=False):
        cuts = tuple(sorted(set(cuts)))
        if cuts not in self.bands:
            bands = defaultdict(list)
            for v in self.graph.order:
                bands[bisect_left(cuts, self.graph.depth[v])].append(v)
            self.bands[cuts] = [self.graph.connected(nodes) for nodes in bands.values()]
        if resident:
            return self.resident_layers(self.bands[cuts], cores)
        return self.dispatch([g for units in self.bands[cuts]
                              for g in self.pack(units, cores, affinity)], cores)

    def resident_layers(self, bands, cores):
        """逐层段联合分核，沿已放置的生产者保持张量亲和，允许核心商图有环。"""
        blocks, owner = [[] for _ in range(cores)], {}
        loads, seen = [[0] * 4 for _ in range(cores)], [set() for _ in range(cores)]
        for units in bands:
            groups = [[] for _ in range(cores)]
            for nodes in sorted(units, key=lambda g: (-max(self.unit(g)[0]), g[0])):
                work, inputs, _ = self.unit(nodes)
                def choice(c):
                    cross = sum(self.records[t][3] * len({owner[p] for p in self.graph.producers[t]} - {c})
                                for t in inputs if self.graph.producers[t])
                    fresh = sum(self.records[t][3] for t in inputs - seen[c] if not self.graph.producers[t])
                    return max(a + b for a, b in zip(loads[c], work)) + 2 * cross + fresh, sum(loads[c]), c
                _, _, c = min(choice(c) for c in range(cores))
                groups[c].extend(nodes)
                loads[c] = [a + b for a, b in zip(loads[c], work)]
                owner.update((v, c) for v in nodes)
                seen[c].update(inputs)
            for c, group in enumerate(groups):
                if group:
                    blocks[c].append(group)
        return Plan.build(blocks)

    def chunks(self, units, count):
        target = sum(sum(self.graph.ops[v]['cycles'] for v in g) for g in units) / count
        groups, current, load = [], [], 0
        for nodes in units:
            if current and load >= target and len(groups) < count - 1:
                groups.append(current)
                current, load = [], 0
            current.extend(nodes)
            load += sum(self.graph.ops[v]['cycles'] for v in nodes)
        if current:
            groups.append(current)
        return groups


    def streaming(self, cores):
        """将相邻阶段分配给不同核心组，允许 COPY 与计算逐张量流水重叠。"""
        graph, cfg = self.graph, self.settings
        depth = max(graph.depth.values(), default=0)
        work, changes = defaultdict(lambda: [0] * 4), defaultdict(int)
        total = [0] * 4
        for v in graph.order:
            p, duration = self.pipe[v], max(1, graph.ops[v]['cycles'])
            work[graph.depth[v]][p] += duration
            total[p] += duration
        for ps, cs, size, _, _ in self.records.values():
            if ps and cs:
                a, b = min(graph.depth[v] for v in ps), max(graph.depth[v] for v in cs)
                if a < b:
                    changes[a] += size
                    changes[b] -= size
        left, crossing, ranked = [0] * 4, 0, []
        for cut in range(1, depth):
            left = [a + b for a, b in zip(left, work[cut])]
            crossing += changes[cut]
            right = [a - b for a, b in zip(total, left)]
            for count in range(1, cores):
                score = max(max(left) / count, max(right) / (cores - count),
                            2 * crossing / cfg.bandwidth)
                ranked.append((score, crossing, cut, count))
        for _, _, cut, count in sorted(ranked)[:3]:
            if (cut,) not in self.bands:
                self.bands[cut,] = [graph.connected([v for v in graph.order
                                   if (graph.depth[v] <= cut) == before]) for before in (True, False)]
            before, after = self.bands[cut,]
            first, last = self.pack(before, count), self.pack(after, cores - count)
            yield f'stream_{cut}_{count}', Plan.build(
                [[g] for g in first] + [[] for _ in range(count - len(first))]
                + [[g] for g in last] + [[] for _ in range(cores - count - len(last))])


    def refinements(self, solution):
        """只重切实际耗时最大的子图；搜索内部依赖边界以暴露被合并的并行分支。"""
        plan, graph = solution.plan, self.graph
        items = [group for core in plan.blocks for group in core]
        busy = defaultdict(lambda: [0] * 4)
        for core in solution.result['per_core_timeline']:
            for op in core['ops']:
                busy[op['subgraph_id']][PIPES.index(op['pipe'])] += op['duration']
        if not busy:
            return
        s = max(busy, key=lambda s: (max(busy[s]), -s))
        nodes = items[s]
        cuts = sorted({graph.depth[v] for v in nodes})[:-1]
        if len(graph.ops) > 4096 and len(cuts) > 8:
            widths = defaultdict(int)
            for v in nodes:
                widths[graph.depth[v]] += 1
            changes = sorted(cuts, key=lambda d: (-abs(widths[d + 1] - widths[d]), d))[:4]
            cuts = sorted(set(changes + cuts[:2] + [cuts[len(cuts) // 2], cuts[-1]]))
        elif len(cuts) > 24:
            cuts = [cuts[round(i * (len(cuts) - 1) / 23)] for i in range(24)]
        for cut in cuts:
            groups = [g for i, g in enumerate(items) if i != s]
            for before in (True, False):
                band = [v for v in nodes if (graph.depth[v] <= cut) == before]
                groups.extend(self.pack(graph.connected(band), len(plan.blocks)))
            try:
                yield f'cut_{s}_{cut}', self.dispatch(groups, len(plan.blocks))
            except ValueError:
                continue

    def candidates(self, cores):
        """全部候选只由原图产生；保留分组以控制同核访问顺序。"""
        graph, depth = self.graph, max(self.graph.depth.values(), default=0)
        yield from self.streaming(cores)
        for scale in (1, 2, 4):
            if len(graph.components) > 1:
                groups = self.pack(graph.components, cores * scale)
                yield f'components_{scale}', self.dispatch(groups, cores)
                if scale == 4 and len(groups) > cores:
                    a, b = sorted(range(len(groups)), key=lambda i: (self.unit(groups[i])[2], groups[i][0]))[:2]
                    merged = [g for i, g in enumerate(groups) if i not in (a, b)] + [groups[a] + groups[b]]
                    yield 'coarsen', self.dispatch(merged, cores)
        if len(graph.components) > 1:
            yield 'shared_inputs', self.dispatch(self.pack(graph.components, cores, 0.5), cores)
        chain_of = {v: i for i, chain in enumerate(self.chains) for v in chain}
        units = [self.chains[i] for i in dict.fromkeys(chain_of[v] for v in self.dfs)]
        for scale in (1, 2, 4):
            yield f'dfs_{scale}', self.dispatch(self.chunks(units, cores * scale), cores)
            yield f'topo_{scale}', self.dispatch(self.chunks([[v] for v in graph.order], cores * scale), cores)
        for stages in (2, 4):
            yield f'levels_{stages}', self.layers(cores, [math.ceil(depth * j / stages)
                                                        for j in range(1, stages)])
        for width in (2, 4, 8, 16):
            if width < depth:
                yield f'window_{width}', self.layers(cores, range(width, depth, width))
                if width >= 4:
                    yield f'resident_{width}', self.layers(cores, range(width, depth, width), resident=True)
        widths = defaultdict(int)
        for d in graph.depth.values():
            widths[d] += 1
        cuts = set()
        for v in graph.order:
            d = graph.depth[v]
            incoming = sum(graph.tensors[t]['size'] for t in graph.inputs[v])
            outgoing = sum(graph.tensors[t]['size'] for t in graph.outputs[v])
            if d < depth and ((len(graph.succs[v]) > 1 and widths[d + 1] >= max(cores, 2 * widths[d]))
                              or (incoming > self.settings.bandwidth and 4 * outgoing <= incoming)):
                cuts.add(d)
        if cuts:
            yield 'fork_reduce', self.layers(cores, cuts, 0.5)
            yield 'resident_forks', self.layers(cores, cuts, resident=True)

"""第三问：张量代价、可证时间下界及从原图生成的多粒度初始方案。"""

from collections import defaultdict, deque
from functools import lru_cache
import math

from tools.graph import topological
from tools.official import PIPES, step1_schedule
from tools.plan import Plan, block_graph
from tools.preview import copy_traffic


class Model:
    def __init__(self, graph, settings):
        self.graph, self.settings = graph, settings
        self.position = {v: i for i, v in enumerate(graph.order)}
        self.pipe = {v: PIPES.index(o['pipe']) for v, o in graph.ops.items()}
        self.work = {v: max(1, o['cycles']) for v, o in graph.ops.items()}
        self.records, self.transfer = {}, defaultdict(dict)
        self.inputs = set()
        for t, tensor in graph.tensors.items():
            ps, cs = graph.producers[t], graph.consumers[t]
            if not ps and not cs:
                continue
            size = tensor['size']
            ddr = max(1, math.ceil(size / settings.bandwidth))
            eligible = 0 < size <= settings.cache_capacity
            fast = min(ddr, max(1, math.ceil(size / settings.cache_bandwidth))) if eligible else ddr
            self.records[t] = (ps, cs, size, ddr, fast, t in graph.final or not cs)
            if cs and not ps:
                self.inputs.add(t)
            for p in ps:
                for c in cs:
                    self.transfer[c][p] = max(self.transfer[c].get(p, 0), ddr + fast)
        self.direct = [(e['source'], e['target'], int(e.get('data_size', 0)))
                       for e in graph.raw['edges']
                       if e['source'] in graph.ops and e['target'] in graph.ops]
        represented = {v: set(self.transfer[v]) for v in graph.ops}
        for u, v, size in self.direct:
            represented[v].add(u)
            q = max(1, math.ceil(size / settings.bandwidth))
            fast = min(q, max(1, math.ceil(size / settings.cache_bandwidth)))
            self.transfer[v][u] = max(self.transfer[v].get(u, 0), q + fast)
        self.path_valid = all(graph.preds[v] <= represented[v] for v in graph.ops)
        self.rank = {}
        for v in reversed(graph.order):
            self.rank[v] = self.work[v] + max((self.rank[w] for w in graph.succs[v]), default=0)
        self.total_work = [sum(self.work[v] for v in graph.ops if self.pipe[v] == p) for p in range(4)]
        # 自定义 logical_tid 别名可能共享冷读取；此时放弃输入部分的下界。
        unique_keys = not any('logical_tid' in t for t in graph.tensors.values())
        self.mandatory = sum(ddr * (int(bool(ps) and final) + int(unique_keys and bool(cs) and not ps))
                             for ps, cs, _, ddr, _, final in self.records.values())
        self.minimum_added = sum(size * (int(bool(ps) and final) + int(bool(cs) and not ps))
                                 for ps, cs, size, _, _, final in self.records.values()) - copy_traffic(graph.raw)
        self.io_floor = 0
        if self.path_valid and unique_keys and graph.ops:
            # 第一个计算前须冷读其全部输入；最后一个计算后须写完其最终输出。
            # 两段均不能与中间的计算区间重叠，且 DDR 总带宽只有一份。
            first = min(sum(self.records[t][3] for t in graph.inputs[v] if not graph.producers[t])
                        for v in graph.order if not graph.preds[v])
            last = min(sum(self.records[t][3] for t in graph.outputs[v] if self.records[t][5])
                       for v in graph.order if not graph.succs[v])
            self.io_floor = first + last
        self.critical = topological(graph.preds, graph.succs, {v: -self.rank[v] for v in graph.ops})
        pressure = {}
        for v in graph.ops:
            def fraction(t):
                tensor = graph.tensors[t]
                space = tensor['pos'] if tensor['pos'] in settings.capacity else 'UB'
                return tensor['size'] / max(1, settings.capacity[space])
            pressure[v] = (sum(fraction(t) for t in graph.outputs[v])
                           - sum(fraction(t) for t in graph.inputs[v] if len(graph.consumers[t]) == 1),
                           -self.rank[v])
        self.release = topological(graph.preds, graph.succs, pressure)
        self.dfs = [v for v in step1_schedule(graph.raw) if v in graph.ops]
        self.chains = graph.chains()
        self._units, self._bands = {}, {}
        self._estimate = lru_cache(maxsize=32)(self._estimate)

    def lower_bound(self, cores):
        return max(math.ceil(max(self.total_work, default=0) / cores) + self.io_floor, self.mandatory,
                   max(self.rank.values(), default=0) if self.path_valid else 0)

    def estimate(self, plan):
        """返回 (安全下界, 必需 COPY 字节)。忽略 Spill，不臆测命中率。"""
        owner = plan.assignment()
        return self._estimate(tuple(owner[v] for v in self.graph.order), len(plan.blocks))

    def _estimate(self, assignment, cores):
        # 此下界只依赖分核；子图分组/顺序的完整合法性仍逐候选检查。
        graph, cfg = self.graph, self.settings
        owner = dict(zip(graph.order, assignment))
        loads, writes, traffic = [[0] * 4 for _ in range(cores)], 0, 0
        for v, c in owner.items():
            loads[c][self.pipe[v]] += self.work[v]
        for ps, cs, size, ddr, fast, final in self.records.values():
            sources, targets = {owner[v] for v in ps}, {owner[v] for v in cs}
            readers = list(targets) if not ps else []
            writers = list(sources) if ps and final else []
            for src in sources:
                for dst in targets - {src}:
                    writers.append(src)
                    readers.append(dst)
            for c in readers:
                loads[c][0] += fast
            for c in writers:
                loads[c][1] += ddr
            writes += ddr * len(writers)
            traffic += size * (len(readers) + len(writers))
        for u, v, size in self.direct:
            if owner[u] != owner[v]:
                q = max(1, math.ceil(size / cfg.bandwidth))
                loads[owner[u]][1] += q
                loads[owner[v]][0] += min(q, max(1, math.ceil(size / cfg.cache_bandwidth)))
                writes += q
                traffic += 2 * size
        bound = max(self.lower_bound(cores), writes, max(map(max, loads), default=0))
        if self.path_valid:
            finish = {}
            for v in graph.order:
                finish[v] = self.work[v] + max((finish[u] + (cfg.copy_delay + self.transfer[v][u]
                    if owner[u] != owner[v] else 0) for u in graph.preds[v]), default=0)
            bound = max(bound, max(finish.values(), default=0))
        return bound, traffic

    def unit(self, nodes, boundary=True):
        nodes = tuple(sorted(nodes))
        key = nodes, boundary
        if key not in self._units:
            g, members = self.graph, set(nodes)
            work, inputs, outputs = [0] * 4, set(), set()
            for v in nodes:
                work[self.pipe[v]] += self.work[v]
                inputs.update(t for t in g.inputs[v] if not g.producers[t] & members)
                outputs.update(t for t in g.outputs[v]
                               if t in g.final or not g.consumers[t] or g.consumers[t] - members)
            read = sum(self.records[t][3] for t in inputs)
            write = sum(self.records[t][3] for t in outputs)
            finish, critical = {}, 0
            for v in sorted(nodes, key=self.position.get):
                ready = max((finish[u] for u in g.preds[v] if u in members), default=0)
                if boundary:
                    ready = max(ready, max((self.records[t][3] for t in g.inputs[v] & inputs), default=0))
                finish[v] = ready + self.work[v]
                drain = max((self.records[t][3] for t in g.outputs[v] & outputs), default=0) if boundary else 0
                critical = max(critical, finish[v] + drain)
            duration = max(max(work), work[0] + read, work[1] + write,
                           read + write, critical)
            self._units[key] = (work, inputs, duration)
        return self._units[key]

    def ordered_bound(self, plan):
        """保留同核同 Pipe 的子图先后关系，不引入整子图完成屏障。

        每个 (子图, Pipe) 增加开始/结束虚点，二者间隔至少为该 Pipe
        总计算量，包围其全部算子；前一组结束到下一组开始相连。
        任意真实执行都满足这些差分约束，其最长路因此是安全下界。
        """
        if not self.path_valid or sum(map(len, plan.blocks)) <= len(plan.blocks):
            return 0
        g, owner = self.graph, plan.assignment()
        early, finish = {}, {}
        def wait(u, v):
            return self.settings.copy_delay + self.transfer[v][u] if owner[u] != owner[v] else 0
        for v in g.order:
            ready = max((self.records[t][4] for t in g.inputs[v] if not g.producers[t]), default=0)
            early[v] = max(ready, max((finish[u] + wait(u, v) for u in g.preds[v]), default=0))
            finish[v] = early[v] + self.work[v]
        successors = [[] for _ in g.order]
        degree, value = [0] * len(g.order), [finish[v] for v in g.order]
        def add(u, v, weight):
            successors[u].append((v, weight))
            degree[v] += 1
        for v in g.order:
            for u in g.preds[v]:
                add(self.position[u], self.position[v], self.work[v] + wait(u, v))
        for core in plan.blocks:
            previous = {}
            for group in core:
                pipes = defaultdict(list)
                for v in group:
                    pipes[self.pipe[v]].append(v)
                for pipe, nodes in pipes.items():
                    start, end = len(successors), len(successors) + 1
                    successors.extend(([], []))
                    degree.extend((0, 0))
                    value.extend((min(early[v] for v in nodes), 0))
                    add(start, end, sum(self.work[v] for v in nodes))
                    for v in nodes:
                        add(start, self.position[v], self.work[v])
                        add(self.position[v], end, 0)
                    if pipe in previous:
                        add(previous[pipe], start, 0)
                    previous[pipe] = end
        ready = deque(i for i, d in enumerate(degree) if not d)
        visited = 0
        while ready:
            u = ready.popleft()
            visited += 1
            for v, weight in successors[u]:
                value[v] = max(value[v], value[u] + weight)
                degree[v] -= 1
                if not degree[v]:
                    ready.append(v)
        if visited != len(value):
            raise ValueError('子图顺序与逐 Pipe 数据依赖形成环')
        return max(value, default=0)

    def dispatch(self, groups, cores, affinity=1.0, boundary=True):
        """边界读写感知的列表分核；代理只生成候选，不用作剪枝下界。"""
        groups = [tuple(sorted(g)) for g in groups if g]
        if boundary:
            groups.sort(key=lambda g: g[0])
        _, preds, succs = block_graph(self.graph, Plan.build([groups]))
        order = topological(preds, succs)
        info = [self.unit(g, boundary) for g in groups]
        group_of = {v: s for s, group in enumerate(groups) for v in group}
        rank = {}
        for s in reversed(order):
            rank[s] = info[s][2] + max((rank[w] for w in succs[s]), default=0)
        available, owner, finish = [0] * cores, {}, {}
        blocks = [[] for _ in range(cores)]
        seen = [set() for _ in range(cores)]
        for s in topological(preds, succs, {s: -rank[s] for s in preds}):
            _, inputs, duration = info[s]
            def cost(c):
                ready = max((finish[p] + (self.settings.copy_delay if owner[p] != c else 0)
                             for p in preds[s]), default=0)
                cross = sum(self.records[t][3] * len({owner[group_of[v]]
                            for v in self.graph.producers[t]} - {c}) for t in inputs)
                end = max(available[c], ready) + duration
                fresh = sum(self.records[t][3] for t in (inputs & self.inputs) - seen[c]) if not boundary else 0
                return end + affinity * (2 * cross + fresh), end, available[c] if boundary else 0, c
            _, end, _, c = min(cost(c) for c in range(cores))
            finish[s] = available[c] = end
            owner[s] = c
            seen[c].update(inputs)
            blocks[c].append(groups[s])
        return Plan.build(blocks)

    def pack(self, units, count, affinity=0.0):
        """合并无相互依赖的分支，形成不同粒度的候选子图。"""
        count = max(1, min(count, len(units)))
        groups, load, seen = [[] for _ in range(count)], [[0] * 4 for _ in range(count)], [set() for _ in range(count)]
        for nodes in sorted(units, key=lambda g: (-max(self.unit(g)[0]), min(g))):
            work, inputs, _ = self.unit(nodes)
            def cost(c):
                return (max(a + b for a, b in zip(load[c], work))
                        + affinity * sum(self.records[t][3] for t in inputs - seen[c]), sum(load[c]), c)
            c = min(range(count), key=cost)
            groups[c].extend(nodes)
            load[c] = [a + b for a, b in zip(load[c], work)]
            seen[c].update(inputs)
        return [g for g in groups if g]

    def bands(self, cuts):
        cuts = tuple(sorted(set(cuts)))
        if cuts not in self._bands:
            from bisect import bisect_left
            nodes = defaultdict(list)
            for v in self.graph.order:
                nodes[bisect_left(cuts, self.graph.depth[v])].append(v)
            self._bands[cuts] = [self.graph.connected(group) for group in nodes.values()]
        return self._bands[cuts]

    def waves(self, cores, cuts, affinity=1.0):
        """每个拓扑层段内并行，跨层保留生产者亲和；不强制核心商图无环。"""
        blocks, owner = [[] for _ in range(cores)], {}
        load, seen = [[0] * 4 for _ in range(cores)], [set() for _ in range(cores)]
        for units in self.bands(cuts):
            groups = [[] for _ in range(cores)]
            for nodes in sorted(units, key=lambda g: (-max(self.unit(g)[0]), min(g))):
                work, inputs, _ = self.unit(nodes)
                def cost(c):
                    cross = sum(self.records[t][3] * len({owner[p] for p in self.graph.producers[t]} - {c})
                                for t in inputs if self.graph.producers[t])
                    fresh = sum(self.records[t][3] for t in (inputs & self.inputs) - seen[c])
                    return max(a + b for a, b in zip(load[c], work)) + 2 * cross + affinity * fresh, sum(load[c]), c
                _, _, c = min(cost(c) for c in range(cores))
                groups[c].extend(nodes)
                owner.update((v, c) for v in nodes)
                load[c] = [a + b for a, b in zip(load[c], work)]
                seen[c].update(inputs)
            for c, group in enumerate(groups):
                if group:
                    blocks[c].append(group)
        return Plan.build(blocks)

    def streaming(self, cores):
        """为前后计算阶段选择核心组；候选排序考虑跨阶段张量流量。"""
        if cores == 1:
            return
        g, total = self.graph, self.total_work
        work, changes = defaultdict(lambda: [0] * 4), defaultdict(int)
        for v in g.order:
            work[g.depth[v]][self.pipe[v]] += self.work[v]
        for ps, cs, size, _, _, _ in self.records.values():
            if ps and cs:
                a, b = min(g.depth[v] for v in ps), max(g.depth[v] for v in cs)
                if a < b:
                    changes[a] += size
                    changes[b] -= size
        left, traffic, options = [0] * 4, 0, []
        for cut in range(1, max(g.depth.values(), default=1)):
            left = [a + b for a, b in zip(left, work[cut])]
            traffic += changes[cut]
            right = [a - b for a, b in zip(total, left)]
            for count in range(1, cores):
                score = max(max(left) / count, max(right) / (cores - count),
                            2 * traffic / self.settings.bandwidth)
                options.append((score, traffic, cut, count))
        for _, _, cut, count in sorted(options)[:3]:
            before, after = self.bands((cut,))
            first, last = self.pack(before, count), self.pack(after, cores - count)
            yield f'stream_{cut}_{count}', Plan.build([[g] for g in first]
                + [[] for _ in range(count - len(first))] + [[g] for g in last]
                + [[] for _ in range(cores - count - len(last))])

    def staged(self, cores):
        """各层段独立打包，再对整体子图 DAG 分核，允许相邻阶段交换核心。"""
        depth = max(self.graph.depth.values(), default=0)
        if depth < 2:
            return
        options = [(f'width_{width}', tuple(range(width, depth, width)))
                   for width in (2, 3, 4, 6, 8, 12, 16, 24, 32) if width < depth]
        widths = defaultdict(int)
        for d in self.graph.depth.values():
            widths[d] += 1
        cuts = (range(1, depth) if depth <= 12 else
                {1, depth // 2, *sorted(range(1, depth),
                 key=lambda d: (-abs(widths[d + 1] - widths[d]), d))[:2]})
        options.extend((f'cut_{cut}', (cut,)) for cut in sorted(cuts))
        seen = set()
        for name, cuts in options:
            if cuts in seen:
                continue
            seen.add(cuts)
            groups = [group for units in self.bands(cuts) for group in self.pack(units, cores)]
            yield f'stage_{name}', self.dispatch(groups, cores, boundary=False)

    def seeds(self, cores):
        g, depth = self.graph, max(self.graph.depth.values(), default=1)
        if not g.ops:
            return
        if len(g.components) > 1:
            for scale in (1, 2, 4):
                groups = self.pack(g.components, cores * scale)
                yield f'components_{scale}', self.dispatch(groups, cores)
            if len(groups) > cores:
                groups = sorted(groups, key=lambda x: (self.unit(x)[2], min(x)))
                yield 'coarsen', self.dispatch([groups[0] + groups[1]] + groups[2:], cores)
            yield 'input_affinity', self.dispatch(self.pack(g.components, cores, 1), cores)
        yield from self.streaming(cores)
        if cores > 1:
            for count in (2, 4):
                cuts = [math.ceil(depth * i / count) for i in range(1, count)]
                groups = [group for units in self.bands(cuts) for group in self.pack(units, cores)]
                yield f'levels_{count}', self.dispatch(groups, cores)
        position = {v: i for i, v in enumerate(self.dfs)}
        chains = sorted(self.chains, key=lambda chain: min(position[v] for v in chain))
        for name, units in (('dfs', chains), ('topo', [[v] for v in g.order])):
            for scale in (1, 2, 4):
                if cores == 1:
                    order = self.dfs if name == 'dfs' else g.order
                    count = min(len(order), scale)
                    groups = [order[len(order) * i // count:len(order) * (i + 1) // count] for i in range(count)]
                else:
                    groups = self.chunks(units, cores * scale)
                yield f'{name}_{scale}', self.dispatch(groups, cores, boundary=cores > 1)
        for width in (2, 4, 8, 16, 32):
            if width < depth:
                yield f'wave_{width}', self.waves(cores, range(width, depth, width))
        widths = defaultdict(int)
        for d in g.depth.values():
            widths[d] += 1
        cuts = {g.depth[v] for v in g.order if g.depth[v] < depth and
                (len(g.succs[v]) > 1 and widths[g.depth[v] + 1] >= max(cores, 2 * widths[g.depth[v]])
                 or sum(g.tensors[t]['size'] for t in g.inputs[v]) > 4 *
                    max(1, sum(g.tensors[t]['size'] for t in g.outputs[v])))}
        if cuts:
            yield 'resident_forks', self.waves(cores, cuts)
            groups = [group for units in self.bands(cuts) for group in self.pack(units, cores, 0.5)]
            yield 'fork_reduce', self.dispatch(groups, cores)

    def chunks(self, units, count):
        """按计算工作量切段；传入安全链时保持链完整，避免按节点数假定等负载。"""
        target = sum(self.work[v] for unit in units for v in unit) / max(1, count)
        groups, current, load = [], [], 0
        for unit in units:
            if current and load >= target and len(groups) < count - 1:
                groups.append(current)
                current, load = [], 0
            current.extend(unit)
            load += sum(self.work[v] for v in unit)
        if current:
            groups.append(current)
        return groups

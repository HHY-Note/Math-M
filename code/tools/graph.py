"""保留依赖 DAG 和张量共享关系，所有遍历均不依赖 ID 连续性。"""

from collections import Counter, defaultdict
import heapq

from tools.official import (
    _build_op_adjacency, _contract_excluded_copy_nodes, validate_graph,
)


def topological(preds, succs, priority=None):
    degree = {v: len(p) for v, p in preds.items()}
    priority = priority or {v: v for v in preds}
    ready = [(priority[v], v) for v, d in degree.items() if not d]
    heapq.heapify(ready)
    order = []
    while ready:
        _, v = heapq.heappop(ready)
        order.append(v)
        for w in sorted(succs[v]):
            degree[w] -= 1
            if not degree[w]:
                heapq.heappush(ready, (priority[w], w))
    if len(order) != len(preds):
        raise ValueError('划分或任务顺序产生了依赖环')
    return order


class Graph:
    def __init__(self, raw):
        validate_graph(raw)
        self.raw = raw
        self.all_ops = {o['id']: o for o in raw['ops']}
        self.ops = {i: o for i, o in self.all_ops.items()
                    if o['op'] not in {'COPY_IN', 'COPY_OUT'}}
        self.tensors = {t['id']: t for t in raw['tensors']}
        self.producers, self.consumers = defaultdict(set), defaultdict(set)
        self.inputs, self.outputs = defaultdict(set), defaultdict(set)
        self.final = set()
        for edge in raw['edges']:
            u, v = edge['source'], edge['target']
            if u in self.ops and v in self.tensors:
                self.producers[v].add(u)
                self.outputs[u].add(v)
            if u in self.tensors and v in self.ops:
                self.consumers[u].add(v)
                self.inputs[v].add(u)
            if (u in self.tensors and v in self.all_ops
                    and self.all_ops[v]['op'] == 'COPY_OUT'):
                self.final.add(u)
        _, full_succs = _build_op_adjacency(raw)
        self.preds, self.succs = _contract_excluded_copy_nodes(
            sorted(self.ops), full_succs)
        self.order = topological(self.preds, self.succs)
        self.depth, self.rank = {}, {}
        for v in self.order:
            self.depth[v] = 1 + max((self.depth[u] for u in self.preds[v]), default=0)
        for v in reversed(self.order):
            self.rank[v] = self.ops[v]['cycles'] + max(
                (self.rank[w] for w in self.succs[v]), default=0)
        self.components = self.connected(self.ops)

    def connected(self, nodes):
        remaining = set(nodes)
        groups = []
        for start in sorted(remaining):
            if start not in remaining:
                continue
            remaining.remove(start)
            stack, members = [start], []
            while stack:
                v = stack.pop()
                members.append(v)
                for w in self.preds[v] | self.succs[v]:
                    if w in remaining:
                        remaining.remove(w)
                        stack.append(w)
            groups.append(tuple(sorted(members)))
        return groups

    def chains(self):
        groups, owner = [], {}
        for v in self.order:
            pred = self.preds[v]
            if len(pred) == 1 and len(self.succs[next(iter(pred))]) == 1:
                group = owner[next(iter(pred))]
            else:
                group = len(groups)
                groups.append([])
            owner[v] = group
            groups[group].append(v)
        return groups

    def features(self):
        counts = Counter(o['op'] for o in self.ops.values())
        kinds = set(counts)
        depth = max(self.depth.values(), default=0)
        if kinds == {'CONV', 'RELU', 'ADD'}:
            family = 'conv_shallow' if depth <= 4 else 'conv_deep'
        elif kinds == {'MATMUL', 'RELU', 'ADD'}:
            family = 'gemm_shallow' if depth <= 4 else 'mlp'
        elif kinds == {'MATMUL', 'ADD'}:
            family = 'gemm_reduce'
        elif kinds == {'MATMUL', 'MUL', 'SIGMOID', 'ADD'}:
            family = 'gated'
        elif kinds == {'MATMUL', 'REDUCE', 'SUB', 'EXP', 'DIV'}:
            family = 'attention'
        elif kinds == {'ADD', 'DIV', 'MUL', 'REDUCE', 'RELU', 'SQRT', 'SUB'}:
            family = 'vector_norm'
        elif kinds == {'ADD', 'REDUCE', 'RELU'}:
            family = 'vector_chain'
        elif kinds == {'ADD', 'RELU'}:
            family = 'vector_elementwise'
        elif kinds == {'ADD', 'DIV', 'EXP', 'MATMUL', 'MUL', 'REDUCE',
                       'RELU', 'SIGMOID', 'SQRT', 'SUB'}:
            family = 'transformer'
        else:
            family = 'mixed'
        work = Counter()
        for op in self.ops.values():
            work[op['pipe']] += op['cycles']
        return dict(compute_ops=len(self.ops), tensors=len(self.tensors),
                    edges=len(self.raw['edges']), depth=depth,
                    width=max(Counter(self.depth.values()).values(), default=0),
                    components=len(self.components),
                    largest_component=max(map(len, self.components), default=0),
                    max_fanout=max(map(len, self.consumers.values()), default=0),
                    max_tensor=max((t['size'] for t in self.tensors.values()), default=0),
                    critical_compute=max(self.rank.values(), default=0),
                    family=family, op_counts=dict(counts), pipe_work=dict(work))

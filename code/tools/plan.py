"""场景 B 的通用提交表示；仅负责划分和顺序合法性，不包含搜索算法。"""

from dataclasses import dataclass

from tools.graph import topological
from tools.official import derive_multicore_plan


@dataclass(frozen=True)
class Plan:
    """blocks[core][subgraph] 保存节点；保留空核心及子图顺序。"""
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
        mapping, schedules, sg = {}, [], 0
        for core in self.blocks:
            order = []
            for group in core:
                order.append(sg)
                mapping.update((str(v), sg) for v in group)
                sg += 1
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
    if not 1 <= len(plan.blocks) <= 5:
        raise ValueError('核心数必须在 1～5 范围内')
    items, preds, succs = block_graph(graph, plan)
    topological(preds, succs)
    for a, after in succs.items():
        if any(items[a][0] == items[b][0] and a > b for b in after):
            raise ValueError('同核子图顺序违反依赖')


def arrange(graph, groups, cores, priority):
    blocks = [[] for _ in range(cores)]
    for core, group in groups:
        if group:
            blocks[core].append(group)
    items, preds, succs = block_graph(graph, Plan.build(blocks))
    ranks = {s: min(priority[v] for v in group) for s, (_, group) in enumerate(items)}
    ordered = [[] for _ in range(cores)]
    for s in topological(preds, succs, ranks):
        c, group = items[s]
        ordered[c].append(group)
    return Plan.build(ordered)


def runs(graph, owner, cores, order, width=256, limit=8192):
    """拓扑序的同核连续段；每段是一个子图，允许核心商图有环。"""
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

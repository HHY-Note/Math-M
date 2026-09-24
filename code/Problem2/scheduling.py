"""固定分核时调整子图顺序，并按实际瓶颈生成局部迁移。"""

from Problem2.partition import Plan, arrange, block_graph, check_plan, runs


def order_candidates(model, plan):
    graph, n, owner = model.graph, len(plan.blocks), plan.assignment()
    for name, order in (('critical_runs', model.critical), ('release_runs', model.release)):
        yield name, runs(graph, owner, n, order, width=32)
    # 对已有子图细分，保留原来的核心分配和子图间顺序。
    for parts in (2, 4):
        for name, order in (('split_critical', model.critical), ('split_release', model.release)):
            position = {v: i for i, v in enumerate(order)}
            blocks = []
            for core in plan.blocks:
                pieces = []
                for group in core:
                    nodes = sorted(group, key=position.get)
                    pieces.extend(nodes[len(nodes) * i // parts:len(nodes) * (i + 1) // parts]
                                  for i in range(parts))
                blocks.append(pieces)
            yield name + ('_4' if parts == 4 else ''), Plan.build(blocks)
        if parts == 2:
            pairs = [(model.unit(a)[2] + model.unit(b)[2], c, i)
                     for c, core in enumerate(plan.blocks) for i, (a, b) in enumerate(zip(core, core[1:]))]
            if pairs:
                _, c, i = min(pairs)
                blocks = [list(core) for core in plan.blocks]
                blocks[c][i:i + 2] = [blocks[c][i] + blocks[c][i + 1]]
                candidate = Plan.build(blocks)
                try:
                    check_plan(graph, candidate)
                    yield 'merge_neighbors', candidate
                except ValueError:
                    pass
            groups = [(c, group) for c, core in enumerate(plan.blocks) for group in core]
            yield 'subgraph_critical', arrange(graph, groups, n, {v: -graph.rank[v] for v in graph.ops})
            merged = Plan.build([[[v for v in graph.order if owner[v] == c]] for c in range(n)])
            try:
                check_plan(graph, merged)
                yield 'merge_per_core', merged
            except ValueError:
                pass  # 核心级有环不影响原多子图方案的合法性。


def relocate(model, plan, nodes, target):
    """迁移一个局部单元，按原图拓扑序切开受影响子图，避免把剩余两端错误合并。"""
    moving, groups = set(nodes), []
    for core, blocks in enumerate(plan.blocks):
        for block in blocks:
            if not moving.intersection(block):
                groups.append((core, block))
                continue
            pieces, previous = [], None
            for v in sorted(block, key=model.position.get):
                destination = target if v in moving else core
                if destination != previous:
                    pieces.append((destination, []))
                    previous = destination
                pieces[-1][1].append(v)
            groups.extend(pieces)
    return arrange(model.graph, groups, len(plan.blocks), model.position)


def move_candidates(model, solution, limit=24):
    """检查负载、搬运量和同步等待三类瓶颈；仅官方评分决定是否接受。"""
    graph, plan = model.graph, solution.plan
    owner = plan.assignment()
    finish = {row['core_id']: max((op['end'] for op in row['ops']), default=0)
              for row in solution.result['per_core_timeline']}
    bottleneck = max(finish, key=finish.get)
    items, _, _ = block_graph(graph, plan)
    block_of = {v: s for s, (_, group) in enumerate(items) for v in group}
    units = []
    for c, group in items:
        if c == bottleneck:
            work = sum(max(1, graph.ops[v]['cycles']) for v in group)
            units.append((work, group))
    units.sort(key=lambda x: (-x[0], x[1]))
    choices = [group for _, group in units[:2]]
    boundary = []
    for chain in model.chains:
        if len({owner[v] for v in chain}) != 1 or len({block_of[v] for v in chain}) != 1:
            continue
        c = owner[chain[0]]
        tensors = set().union(*(graph.inputs[v] | graph.outputs[v] for v in chain))
        cut = sum(graph.tensors[t]['size'] for t in tensors
                  if any(owner[v] != c for v in graph.producers[t] | graph.consumers[t]))
        work = sum(max(1, graph.ops[v]['cycles']) for v in chain)
        boundary.append((cut, work if c == bottleneck else 0, tuple(chain)))
    boundary.sort(key=lambda x: (-x[0], -x[1], x[2]))
    choices.extend(group for _, _, group in boundary[:4])
    proposals = []
    for nodes in dict.fromkeys(tuple(sorted(g)) for g in choices):
        source = owner[nodes[0]]
        for target in range(len(plan.blocks)):
            if target != source:
                proposals.append(('load', nodes, target))
    # 小张量也可能引入长同步等待；从关键路径和实际末尾算子寻找跨核边。
    end = {op['op_id']: op['end'] for row in solution.result['per_core_timeline']
           for op in row['ops'] if op['op_id'] in graph.ops}
    edges = [(v, w) for v in graph.order for w in sorted(graph.succs[v]) if owner[v] != owner[w]]
    critical = sorted(edges, key=lambda e: (-graph.rank[e[0]], e))[:4]
    late = sorted(edges, key=lambda e: (-end.get(e[1], 0), e))[:4]
    chains = {v: tuple(chain) for chain in model.chains for v in chain}
    for v, w in dict.fromkeys(critical + late):
        for node, target in ((v, owner[w]), (w, owner[v])):
            nodes = chains[node]
            if len({owner[u] for u in nodes}) == 1:
                proposals.append(('sync', nodes, target))
    proposals = list({(tuple(sorted(nodes)), target): (kind, nodes, target)
                      for kind, nodes, target in proposals}.values())
    if len(graph.ops) > 4096:
        loads = [[0] * 4 for _ in plan.blocks]
        for v, c in owner.items():
            loads[c][model.pipe[v]] += max(1, graph.ops[v]['cycles'])
        def hint(item):
            _, nodes, target = item
            source, work = owner[nodes[0]], model.unit(nodes)[0]
            peak = max(max(value + (work[p] if c == target else -work[p] if c == source else 0)
                           for p, value in enumerate(row)) for c, row in enumerate(loads))
            members = set(nodes)
            tensors = set().union(*(graph.inputs[v] | graph.outputs[v] for v in nodes))
            cut = sum(model.records[t][3] * len({owner[v] for v in
                      (graph.producers[t] | graph.consumers[t]) - members} - {target}) for t in tensors)
            return peak + 2 * cut, nodes[0], target
        proposals = [item for kind in ('load', 'sync') for item in
                     sorted((p for p in proposals if p[0] == kind), key=hint)[:max(4, limit)]]
    seen, ranked = {plan}, []
    for kind, nodes, target in proposals:
        source = owner[nodes[0]]
        try:
            candidate = relocate(model, plan, nodes, target)
        except ValueError:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        ranked.append((model.estimate(candidate), f'{kind}_{source}_to_{target}_{nodes[0]}', candidate))
    ranked.sort(key=lambda item: (item[0], item[1]))
    # 分配一半名额给同步候选，避免单纯搬运量排序排除有用的小张量迁移。
    chosen = [item for item in ranked if item[1].startswith('sync_')][:limit // 2]
    chosen.extend(item for item in ranked if item not in chosen)
    return [(name, plan) for _, name, plan in chosen[:limit]]

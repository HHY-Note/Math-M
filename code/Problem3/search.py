"""第三问的顺序、瓶颈重切和 Cache 访问反馈邻域；全部提案须精确评测。"""

from collections import Counter, defaultdict

from tools.plan import Plan, arrange, block_graph, runs


def split_order(plan, order, parts=2):
    position = {v: i for i, v in enumerate(order)}
    blocks = []
    for core in plan.blocks:
        pieces = []
        for group in core:
            nodes = sorted(group, key=position.get)
            pieces.extend(nodes[len(nodes) * i // parts:len(nodes) * (i + 1) // parts] for i in range(parts))
        blocks.append(pieces)
    return Plan.build(blocks)


def orders(model, plan):
    owner, cores = plan.assignment(), len(plan.blocks)
    for name, order in (('critical', model.critical), ('release', model.release), ('dfs', model.dfs)):
        yield f'runs_{name}', runs(model.graph, owner, cores, order, width=32)
    for parts in (2, 4):
        for name, order in (('critical', model.critical), ('release', model.release)):
            yield f'split_{name}_{parts}', split_order(plan, order, parts)
    groups = [(c, group) for c, core in enumerate(plan.blocks) for group in core]
    yield 'block_critical', arrange(model.graph, groups, cores, {v: -model.rank[v] for v in owner})
    adjacent = [(model.unit(a)[2] + model.unit(b)[2], c, i)
                for c, core in enumerate(plan.blocks) for i, (a, b) in enumerate(zip(core, core[1:]))]
    if adjacent:
        _, c, i = min(adjacent)
        blocks = [list(core) for core in plan.blocks]
        blocks[c][i:i + 2] = [blocks[c][i] + blocks[c][i + 1]]
        yield 'merge_adjacent', Plan.build(blocks)
    yield 'merge_per_core', Plan.build([[[v for group in core for v in group]] for core in plan.blocks])


def relocate(model, plan, nodes, target):
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


def recut(model, items, region, cores, cut, boundary=True):
    nodes = [v for s in region for v in items[s][1]]
    groups = [group for s, (_, group) in enumerate(items) if s not in region]
    for before in (True, False):
        band = [v for v in nodes if (model.graph.depth[v] <= cut) == before]
        groups.extend(model.pack(model.graph.connected(band), cores))
    return model.dispatch(groups, cores, boundary=boundary)


def refinements(model, solution):
    """按最忙 Pipe 定位子图，保留小图全部切点供下界排序。"""
    g, plan, result = model.graph, solution.plan, solution.result
    cores = len(plan.blocks)
    if cores == 1:
        return
    items, _, _ = block_graph(g, plan)
    busy = defaultdict(Counter)
    for core in result['per_core_timeline']:
        for op in core['ops']:
            busy[op['subgraph_id']][op['pipe']] += op['duration']
    if busy:
        s = max(busy, key=lambda s: (max(busy[s].values()), -s))
        nodes = items[s][1]
        depths = sorted({g.depth[v] for v in nodes})[:-1]
        if depths:
            counts = Counter(g.depth[v] for v in nodes)
            cuts = set(depths) if len(depths) <= 24 else {
                depths[round(i * (len(depths) - 1) / 23)] for i in range(24)}
            if len(g.ops) > 4096 and len(cuts) > 8:
                cuts = {depths[0], depths[1], depths[len(depths) // 2], depths[-1],
                        *sorted(cuts, key=lambda d: (-abs(counts[d + 1] - counts[d]), d))[:4]}
            for cut in sorted(cuts):
                try:
                    yield f'recut_{s}_{cut}', recut(model, items, {s}, cores, cut)
                except ValueError:
                    continue


def moves(model, solution):
    """对排序后的实际时间线做负载迁移与关键跨核同步消除。"""
    g, plan, result = model.graph, solution.plan, solution.result
    cores, owner = len(plan.blocks), plan.assignment()
    if cores == 1:
        return
    items, _, _ = block_graph(g, plan)
    finish = {row['core_id']: max((op['end'] for op in row['ops']), default=0)
              for row in result['per_core_timeline']}
    slow = max(finish, key=finish.get)
    groups = sorted((group for c, group in items if c == slow),
                    key=lambda group: (-sum(model.work[v] for v in group), min(group)))[:2]
    proposals = [('load', group, c) for group in groups for c in range(cores) if c != slow]
    chains = {v: tuple(chain) for chain in model.chains for v in chain}
    subgraph = {v: s for s, (_, group) in enumerate(items) for v in group}
    crossing = []
    for group in model.chains:
        if len({owner[v] for v in group}) > 1 or len({subgraph[v] for v in group}) > 1:
            continue
        tensors = set().union(*(g.inputs[v] | g.outputs[v] for v in group))
        traffic = sum(model.records[t][2] for t in tensors
                      if len({owner[v] for v in g.producers[t] | g.consumers[t]}) > 1)
        work = sum(model.work[v] for v in group) if owner[group[0]] == slow else 0
        crossing.append((traffic, work, tuple(group)))
    for _, _, group in sorted(crossing, key=lambda x: (-x[0], -x[1], x[2]))[:4]:
        proposals.extend(('load', group, c) for c in range(cores) if c != owner[group[0]])
    boundary = [(v, w) for v in g.order for w in sorted(g.succs[v]) if owner[v] != owner[w]]
    end = {op['op_id']: op['end'] for row in result['per_core_timeline'] for op in row['ops']
           if op['op_id'] in g.ops}
    critical = sorted(boundary, key=lambda e: (-model.rank[e[0]], e))[:4]
    late = sorted(boundary, key=lambda e: (-end.get(e[1], 0), e))[:4]
    for v, w in dict.fromkeys(critical + late):
        for node, target in ((v, owner[w]), (w, owner[v])):
            group = chains[node]
            if len({owner[u] for u in group}) == 1:
                proposals.append(('sync', group, target))
    seen = set()
    for kind, nodes, target in proposals:
        key = tuple(sorted(nodes)), target
        if key in seen:
            continue
        seen.add(key)
        try:
            yield f'{kind}_{min(nodes)}_{target}', relocate(model, plan, nodes, target)
        except ValueError:
            continue


def coarse_neighbors(model, solution):
    """合并同核的局部划分后重新切分，跨越仅重切单个子图的局部极小值。"""
    plan, g = solution.plan, model.graph
    cores = len(plan.blocks)
    if cores == 1:
        return
    items, _, _ = block_graph(g, plan)
    for core in range(cores):
        region = {s for s, (c, _) in enumerate(items) if c == core}
        if len(region) < 2:
            continue
        counts = Counter(g.depth[v] for s in region for v in items[s][1])
        depths = sorted(counts)[:-1]
        if not depths:
            continue
        cuts = set(depths) if len(depths) <= 16 else {
            depths[round(i * (len(depths) - 1) / 5)] for i in range(6)}
        cuts.update(sorted(depths, key=lambda d: (-abs(counts[d + 1] - counts[d]), d))[:2])
        for cut in sorted(cuts):
            try:
                yield f'coarse_{core}_{cut}', recut(model, items, region, cores, cut, boundary=False)
            except ValueError:
                continue


def cache_neighbors(model, solution):
    """针对真实重复冷未命中和淘汰后重读，调整消费子图相位及归属。"""
    g, plan, result = model.graph, solution.plan, solution.result
    owner, cores = plan.assignment(), len(plan.blocks)
    items = [(c, i, group) for c, core in enumerate(plan.blocks) for i, group in enumerate(core)]
    seen, inserted, pressure = set(), set(), Counter()
    affected = defaultdict(set)
    op_subgraph = {(row['core_id'], op['op_id']): op['subgraph_id']
                   for row in result['per_core_timeline'] for op in row['ops']}
    for event in result['cache_events']:
        t = event['tensor_id']
        if event['event'] == 'insert':
            inserted.add(t)
        elif event['event'] == 'miss':
            if t in seen or t in inserted:
                pressure[t] += event['size_bytes']
                sg = op_subgraph.get((event['core_id'], event['op_id']))
                if sg is not None:
                    affected[t].add(sg)
            seen.add(t)
    for t, _ in pressure.most_common(4):
        for sg in sorted(affected[t])[:3]:
            c, index, group = items[sg]
            # 微调实际发生未命中的读取所在子图；合法性由完整检查决定。
            for offset in (-2, -1, 1, 2):
                target = max(0, min(len(plan.blocks[c]) - 1, index + offset))
                if target == index:
                    continue
                blocks = [list(core) for core in plan.blocks]
                blocks[c].insert(target, blocks[c].pop(index))
                yield f'cache_phase_{t}_{c}_{index}_{target}', Plan.build(blocks)
        if t in g.consumers and cores > 1:
            consumers = g.consumers[t]
            counts = Counter(owner[v] for v in consumers)
            if len(counts) > 1:
                target = min(counts, key=lambda c: (-counts[c], c))
                for source in sorted(counts):
                    if source == target:
                        continue
                    nodes = [v for v in consumers if owner[v] == source]
                    try:
                        yield f'cache_gather_{t}_{source}_{target}', relocate(model, plan, nodes, target)
                    except ValueError:
                        continue
    # 同核重读也可能需要较大范围的重排；依据真实热点优先级重建合法拓扑序。
    if pressure:
        priorities = {v: (-sum(pressure[t] for t in g.inputs[v]), -model.rank[v]) for v in g.ops}
        from tools.graph import topological
        order = topological(g.preds, g.succs, priorities)
        yield 'cache_hot_runs', runs(g, owner, cores, order, width=32)

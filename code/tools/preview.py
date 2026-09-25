"""场景 B 两种配置的候选精评；最终均由未修改的官方入口完整核验。

Step3 的四个 pipe_cursor 各自在算子完成时增加一次，其和恰好是完成数。
用这个计数替代每事件扫描所有状态，其他计算原样使用官方函数。
执行图只追加边，因此复制边列表即可；连续两次只读图索引复用同一结果。
函数仅在独立全局字典中克隆，不改官方文件、模块或配置。
"""

import ast
import inspect
import gc
import pickle
from collections import OrderedDict, defaultdict, deque
from types import FunctionType

from tools import official  # 设置原题模块路径。
import schedule_step3 as step3
import multicore_cut_evaluate_problem_2 as scene_b
import multicore_cut_evaluate_problem_3 as scene_cache


def clone(function, **replacements):
    scope = dict(function.__globals__, **replacements)
    return FunctionType(function.__code__, scope, function.__name__,
                        function.__defaults__, function.__closure__)


def cached_views():
    previous, value = None, None
    def views(ops, edges):
        nonlocal previous, value
        if previous is None or previous[0] is not ops or previous[1] is not edges:
            previous, value = (ops, edges), step3._build_graph_views(ops, edges)
        return value
    return views


def copy_traffic(graph):
    copies = {op['id']: op['op'] for op in graph['ops'] if op['op'] in ('COPY_IN', 'COPY_OUT')}
    sizes = {t['id']: t['size'] for t in graph.get('tensors', [])}
    return sum(sizes.get(e['target'], 0) if copies.get(e['source']) == 'COPY_IN'
               else sizes.get(e['source'], 0) if copies.get(e['target']) == 'COPY_OUT' else 0
               for e in graph['edges'])


class _OmitInternalLogs(ast.NodeTransformer):
    """多核输出不引用这些核内日志；不省略任何调度状态或契约检查。"""
    def visit_Expr(self, node):
        call = node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == 'append' and isinstance(call.func.value, ast.Name)
                and call.func.value.id in {'memory_events', 'ddr_contention_log', 'stages'}):
            return None
        return self.generic_visit(node)


source = inspect.getsource(step3.step3_simulation)
old = "all(status == 'done' for status in op_status.values())"
if (source.count(old) != 1 or source.count('deepcopy(') != 1
        or source.count('credits.pop(0)') != 1):
    raise RuntimeError('官方 Step3 完成检查发生变化，需重新验证候选评估适配器')
scope = dict(vars(step3))
views = cached_views()
scope.update(_build_graph_views=views, deque=deque,
             deepcopy=lambda graph: dict(graph, edges=list(graph['edges'])))
source = (source.replace(old, 'sum(pipe_cursor.values()) == len(op_status)')
          .replace('free_credits = {pos: [] for pos in capacity}',
                   'free_credits = {pos: deque() for pos in capacity}')
          .replace('credits.pop(0)', 'credits.popleft()'))
tree = ast.fix_missing_locations(_OmitInternalLogs().visit(ast.parse(source)))
exec(compile(tree, '<scene_b_preview_step3>', 'exec'), scope)
prepare = clone(step3.prepare_step3_execution, step3_simulation=scope['step3_simulation'],
                _build_graph_views=views)
build = clone(scene_b._build_scene_b_tasks, prepare_step3_execution=prepare, _copy_traffic_bytes=copy_traffic)
evaluate_scene_b = clone(scene_b.evaluate_scene_b, _build_scene_b_tasks=build)
build_cache = clone(scene_cache._build_scene_b_tasks, prepare_step3_execution=prepare,
                    _copy_traffic_bytes=copy_traffic)
evaluate_scene_cache = clone(scene_cache.evaluate_problem_3, _build_scene_b_tasks=build_cache)


class Evaluator:
    """单次求解会话内缓存完全相同的核内准备；完整字节键，无哈希碰撞假设。"""
    def __init__(self, raw, settings):
        self.raw, self.settings = raw, settings
        self.entries = OrderedDict()
        self.simulations = OrderedDict()
        self.weight = self.hits = self.misses = 0
        self.simulation_hits = 0
        self.limit = max(4096, min(32768, 2 * len(raw['ops'])))
        self.simulation_limit = 1 if len(raw['ops']) > 8192 else 4
        self.builder = clone(build_cache, prepare_step3_execution=self.prepare)

    def prepare(self, graph, capacity, bandwidth):
        key = pickle.dumps((graph, capacity, bandwidth), protocol=5)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return dict(self.entries[key])
        self.misses += 1
        value = prepare(graph, capacity=capacity, bandwidth=bandwidth)
        value['_preview_key'] = key
        weight = max(1, len(graph['ops']))
        while self.entries and self.weight + weight > self.limit:
            _, old = self.entries.popitem(last=False)
            self.weight -= max(1, len(old['seq']))
        if weight <= self.limit:
            self.entries[key] = value
            self.weight += weight
        # 官方 Task 构造会写入顶层的 core_id / op_subgraph / pipe_cursor。
        return dict(value)

    def simulate(self, raw, plan, bandwidth, capacity, delay, cache_capacity, cache_bandwidth):
        built = self.builder(raw, plan, bandwidth, capacity)
        tasks = built[0]
        # 包含完整扩展图、顺序、容量、带宽、跨核连接和搬运统计。
        # 仅子图标签不参与事件模拟；命中后必须重建这些输出元数据。
        key = (tuple(task['_preview_key'] for task in tasks.values()),
               pickle.dumps((built[1:4], delay, cache_capacity, cache_bandwidth), protocol=5))
        if key in self.simulations:
            self.simulation_hits += 1
            self.simulations.move_to_end(key)
            result = dict(self.simulations[key], per_core_timeline=[])
            for row in self.simulations[key]['per_core_timeline']:
                task = tasks[row['core_id']]
                ops = [dict(op, subgraph_id=task['op_subgraph'].get(op['op_id'])) for op in row['ops']]
                grouped = defaultdict(list)
                for op in ops:
                    grouped[op['subgraph_id']].append(op)
                subgraphs = []
                for sg in task['subgraph_ids']:
                    start = min(op['start'] for op in grouped[sg])
                    end = max(op['end'] for op in grouped[sg])
                    subgraphs.append(dict(subgraph_id=sg, start=start, end=end, duration=end - start))
                result['per_core_timeline'].append(dict(row, ops=ops, subgraphs=subgraphs,
                    tasks=[dict(row['tasks'][0], subgraph_ids=task['subgraph_ids'])]))
            return result
        function = clone(evaluate_scene_cache, _build_scene_b_tasks=lambda *args: built)
        result = function(raw, plan, bandwidth, capacity, delay, cache_capacity, cache_bandwidth)
        self.simulations[key] = result
        if len(self.simulations) > self.simulation_limit:
            self.simulations.popitem(last=False)
        return result

    def __call__(self, plan):
        return evaluate(self.raw, plan, self.settings, cache=True, _function=self.simulate)


def evaluate(raw, plan, settings, *, cache=False, _function=None):
    # 大量短命的图索引主要由引用计数释放，避免构造中反复扫描循环引用。
    enabled = gc.isenabled()
    if enabled:
        gc.disable()
    try:
        if cache:
            return (_function or evaluate_scene_cache)(
                raw, plan, settings.bandwidth, settings.capacity,
                settings.copy_delay, settings.cache_capacity, settings.cache_bandwidth)
        return evaluate_scene_b(raw, plan, settings.bandwidth, settings.capacity, settings.copy_delay)
    finally:
        if enabled:
            gc.enable()
            gc.collect(0)

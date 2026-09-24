"""等价的场景 B 候选评估；最终结果仍须由未修改的官方入口核验。

Step3 的四个 pipe_cursor 各自在算子完成时增加一次，其和恰好是完成数。
用这个计数替代每事件扫描所有状态，其他计算原样使用官方函数。
执行图只追加边，因此复制边列表即可；连续两次只读图索引复用同一结果。
函数仅在独立全局字典中克隆，不改官方文件、模块或配置。
"""

import inspect
import gc
from types import FunctionType

from tools import official  # 设置原题模块路径。
import schedule_step3 as step3
import multicore_cut_evaluate_problem_2 as scene_b


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


source = inspect.getsource(step3.step3_simulation)
old = "all(status == 'done' for status in op_status.values())"
if source.count(old) != 1 or source.count('deepcopy(') != 1:
    raise RuntimeError('官方 Step3 完成检查发生变化，需重新验证候选评估适配器')
scope = dict(vars(step3))
views = cached_views()
scope.update(_build_graph_views=views,
             deepcopy=lambda graph: dict(graph, edges=list(graph['edges'])))
exec(compile(source.replace(old, 'sum(pipe_cursor.values()) == len(op_status)'),
             '<scene_b_preview_step3>', 'exec'), scope)
prepare = clone(step3.prepare_step3_execution, step3_simulation=scope['step3_simulation'],
                _build_graph_views=views)
build = clone(scene_b._build_scene_b_tasks, prepare_step3_execution=prepare, _copy_traffic_bytes=copy_traffic)
evaluate_scene_b = clone(scene_b.evaluate_scene_b, _build_scene_b_tasks=build)


def evaluate(raw, plan, settings):
    # 大量短命的图索引主要由引用计数释放，避免构造中反复扫描循环引用。
    enabled = gc.isenabled()
    if enabled:
        gc.disable()
    try:
        return evaluate_scene_b(raw, plan, settings.bandwidth, settings.capacity, settings.copy_delay)
    finally:
        if enabled:
            gc.enable()
            gc.collect(0)

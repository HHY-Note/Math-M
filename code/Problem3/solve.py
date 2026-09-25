"""问题三独立入口：原图 -> 候选与反馈搜索 -> 提交 JSON -> 原版官方双配置核验。"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import math
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.graph import Graph
from tools.official import CacheSettings, PIPES, read_json, write_json
from tools.plan import Plan, check_plan
from tools.preview import Evaluator
from tools.experiment import digest
from Problem3.model import Model
from Problem3.search import orders, split_order, refinements, moves, cache_neighbors, coarse_neighbors
from Problem3.diagnostics import compact
from evaluation_validation import EvaluationValidationError
from multicore_cut_evaluate_problem_3 import SceneBEvaluationError
from schedule_step2 import Step2SchedulingError
from schedule_step3 import Step3SchedulingError
from stub_multicore_cut_and_schedule import MulticoreCutError

EVALUATION_ERRORS = (ValueError, EvaluationValidationError, SceneBEvaluationError,
                     Step2SchedulingError, Step3SchedulingError, MulticoreCutError)


@dataclass(frozen=True)
class Solution:
    plan: Plan
    result: dict
    origin: str

    @property
    def score(self):
        return self.result['makespan'], self.result['data_movement_bytes']['added_copy_bytes']

    def padded(self, cores):
        """尾部空核无操作、无事件、无带宽占用；已有事件顺序完全不变。"""
        plan = self.plan.padded(cores)
        result = dict(self.result, num_cores=cores, task_count=cores)
        result['per_core_timeline'] = list(self.result['per_core_timeline'])
        for field in ('memory_peak_by_core', 'step3_by_core'):
            result[field] = dict(self.result[field])
        for core in range(len(self.plan.blocks), cores):
            result['memory_peak_by_core'][core] = dict.fromkeys(result['capacity_bytes'], 0)
            result['step3_by_core'][core] = dict(local_makespan=0, memory_dependency_count=0,
                                               pipe_op_counts=dict.fromkeys(PIPES, 0))
            result['per_core_timeline'].append(dict(core_id=core, subgraphs=[], ops=[],
                tasks=[dict(task_id=core, subgraph_ids=[], start=0, end=0, duration=0)]))
        return Solution(plan, result, f'carry_{len(self.plan.blocks)}')


class Solver:
    def __init__(self, raw, settings):
        self.graph = Graph(raw)
        self.settings = settings
        self.model = Model(self.graph, settings)
        self.evaluate = Evaluator(raw, settings)
        self.known = {}
        self._whole = None

    @property
    def whole(self):
        """原图整图单核基准；两种配置都实际调用官方函数。"""
        if self._whole is None:
            plan = Plan.build([[self.graph.order]])
            result = self.settings.evaluate(self.graph.raw, plan.as_dict())
            without = self.settings.without_cache(self.graph.raw, plan.as_dict())
            self._whole = Solution(plan, result, 'whole'), without
        return self._whole

    def solve(self, cores, budget=45.0, seeds=18, local=32, cache=16, rounds=2):
        if type(cores) is not int or cores not in range(1, 6):
            raise ValueError('核心数必须为 1～5 的整数')
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError('搜索时间预算必须为有限正数')
        if any(type(x) is not int or x < 0 for x in (seeds, local, cache, rounds)):
            raise ValueError('候选数及轮数必须为非负整数')
        start = time.perf_counter()
        whole, _ = self.whole
        setup_seconds = time.perf_counter() - start
        started = time.perf_counter()
        old = min([whole] + [s for n, s in self.known.items() if n <= cores], key=lambda s: s.score)
        best = old.padded(cores)
        carried = best.score[0]
        frontier, attempted, history = [best], {best.plan}, []
        coarse_seed = None
        lower = self.model.lower_bound(cores)
        if lower > best.score[0]:
            raise AssertionError('全局下界超过可行解')

        def elapsed():
            return time.perf_counter() - started

        def optimal():
            return best.score == (lower, self.model.minimum_added)

        def consider(plan, name, stage, bound):
            nonlocal best, coarse_seed
            attempted.add(plan)
            before = time.perf_counter()
            entry = dict(stage=stage, candidate=name, lower_bound=bound, called=False)
            try:
                if bound <= best.score[0]:
                    bound = max(bound, self.model.ordered_bound(plan))
                    entry['lower_bound'] = bound
                if bound > best.score[0]:
                    entry['status'] = 'pruned'
                    return
                entry['called'] = True
                result = self.evaluate(plan.as_dict())
                value = Solution(plan, result, name)
                if name == 'levels_2':
                    coarse_seed = value
                if bound > value.score[0]:
                    raise AssertionError(f'候选下界错误: {bound} > {value.score[0]}')
                entry.update(status='evaluated', makespan=value.score[0], added_bytes=value.score[1])
                if value.score < best.score:
                    best = value
                    entry['improved'] = True
                pool = sorted(frontier + [value], key=lambda s: s.score)
                # 最优完整方案始终保留；额外保留不同分核及可继续重切的粗粒度解。
                selected = pool[:3] if cores == 1 else []
                assignments = {tuple(sorted(s.plan.assignment().items())) for s in selected}
                for s in pool:
                    assignment = tuple(sorted(s.plan.assignment().items()))
                    if assignment not in assignments and len(selected) < 3:
                        selected.append(s)
                        assignments.add(assignment)
                eligible = [s for s in pool if s.score[0] <= best.score[0] * 1.5]
                if eligible:
                    coarse = min(eligible, key=lambda s: (sum(map(len, s.plan.blocks)), s.score))
                    light = min(eligible, key=lambda s: (s.score[1], s.score[0]))
                    for s in (coarse, light):
                        if s not in selected:
                            selected.append(s)
                frontier[:] = selected
            except EVALUATION_ERRORS as error:
                entry.update(status='rejected', reason=str(error)[:180])
            finally:
                entry['seconds'] = time.perf_counter() - before
                history.append(entry)

        def run(proposals, stage, limit, fraction):
            deadline = budget * fraction
            if limit == 0 or optimal() or elapsed() >= deadline:
                return 0
            ranked, seen = [], set(attempted)
            for name, plan in proposals:
                if elapsed() >= deadline:
                    break
                if plan is None or plan in seen:
                    continue
                seen.add(plan)
                try:
                    check_plan(self.graph, plan)
                    score = self.model.estimate(plan)
                    ranked.append((score, name, plan))
                except EVALUATION_ERRORS as error:
                    history.append(dict(stage=stage, candidate=name, status='invalid',
                                        called=False, reason=str(error)[:180], seconds=0))
            # 初始候选的相近计算下界会掩盖大量搬运；仅在评测顺序中
            # 加入轻量通信代价，剪枝与最终选优仍使用原始安全下界和精评。
            def priority(item):
                (bound, traffic), name, plan = item
                fine = -sum(map(len, plan.blocks)) if name.split('_')[0] in ('components', 'dfs', 'topo') else 0
                return (bound + (traffic / (10 * self.settings.bandwidth)
                                 if stage in ('seed', 'stage') else 0), traffic, fine, name)
            ranked.sort(key=priority)
            # 每个算子族先获得一次机会，其余按下界/通信量排序，避免单一代理垄断。
            first, rest, families = [], [], set()
            for item in ranked:
                family = item[1].split('_')[0]
                if family in ('runs', 'split'):
                    family = '_'.join(item[1].split('_')[:2])
                (rest if family in families else first).append(item)
                families.add(family)
            calls = 0
            ranked = first + rest
            if stage == 'seed' and cores > 1:
                depth = max(self.graph.depth.values(), default=0)
                primary = ('fork_reduce' if len(self.graph.ops) > 4096 and depth > 32 and
                           len(self.graph.ops) >= depth * len(self.graph.components) / 2 else
                           'components_1' if len(self.graph.components) >= cores else 'levels_2')
                ranked.sort(key=lambda item: (item[1] != primary,
                            not (primary == 'fork_reduce' and item[1] == 'forkrelease')))
            for (bound, _), name, plan in ranked:
                if calls >= limit or optimal() or elapsed() >= deadline:
                    break
                consider(plan, name, stage, bound)
                calls += int(history[-1]['called'])
            return calls

        reserve = min(4, seeds // 3) if cores > 1 else 0
        def initial():
            for name, plan in self.model.seeds(cores):
                yield name, plan
                if cores > 1 and name.startswith('dfs_'):
                    yield 'dfsrelease_' + name[4:], split_order(plan, self.model.release)
                if cores > 1 and name == 'fork_reduce' and len(self.graph.ops) > 4096:
                    yield 'forkrelease', split_order(plan, self.model.release)
        run(initial(), 'seed', seeds - reserve,
            0.50 if reserve or local or cache else 1.0)
        classic = best.score[0]
        if reserve:
            run(self.model.staged(cores), 'stage', reserve, 0.70 if local or cache else 1.0)
        seeded = best.score[0]
        if local:
            refine_limit = min(6, local // 5) if cores > 1 else 0
            move_limit = local // 4 if cores > 1 else 0
            for iteration in range(2):
                base = (coarse_seed or min(frontier, key=lambda s: (sum(map(len, s.plan.blocks)), s.score))
                        if iteration == 0 and len(self.graph.ops) <= 4096 else best)
                run(refinements(self.model, base), 'refine', refine_limit // 2,
                    0.76 if cache else 0.8)
            def structural():
                for index, solution in enumerate(list(frontier)):
                    for name, plan in orders(self.model, solution.plan):
                        yield f'{name}_b{index}', plan
            run(structural(), 'structure', local - refine_limit - move_limit,
                (0.86 if reserve else 0.78) if cache else 0.9)
            def migrations():
                for index, solution in enumerate(list(frontier)[:2]):
                    for name, plan in moves(self.model, solution):
                        yield f'{name}_b{index}', plan
            run(migrations(), 'move', move_limit, 0.94 if cache else 1.0)
        structured = best.score[0]
        for iteration in range(rounds):
            if not cache or elapsed() >= budget:
                break
            before = best.plan
            def feedback():
                for index, solution in enumerate(list(frontier)[:2]):
                    for name, plan in cache_neighbors(self.model, solution):
                        yield f'{name}_r{iteration}_b{index}', plan
            run(feedback(), 'cache', math.ceil(cache / max(1, rounds)), 1.0)
            if best.plan == before:
                break
        cached = best.score[0]
        # 显著改进后瓶颈区域已经改变；用有限的粗化重切跳出原划分的邻域。
        if local and best.score[0] < 0.95 * seeded:
            run(coarse_neighbors(self.model, best), 'polish', max(1, local // 2), 1.0)
        self.known[cores] = best
        report = dict(independent=True, cores=cores, origin=best.origin, carry_makespan=carried,
                      initial_makespan=seeded, structural_makespan=structured, final_makespan=best.score[0],
                      classic_makespan=classic, certified_optimal=optimal(), minimum_added_bytes=self.model.minimum_added,
                      cache_makespan=cached,
                      global_lower_bound=lower,
                      gap_upper_bound=best.score[0] / lower - 1 if lower else None,
                      search_seconds=elapsed(), setup_seconds=setup_seconds,
                      budget_seconds=budget, budget_exhausted=elapsed() >= budget,
                      preview_calls=sum(r['called'] for r in history),
                      preparation_hits=self.evaluate.hits, preparation_misses=self.evaluate.misses,
                      simulation_hits=self.evaluate.simulation_hits,
                      rejected=sum(r['status'] in ('rejected', 'invalid') for r in history),
                      pruned=sum(r['status'] == 'pruned' for r in history), history=history)
        return best, report


def save_verified(solver, solution, output):
    """先导出两字段提交文件，再读取该文件进行原版双配置完整核验。"""
    started = time.perf_counter()
    write_json(output, solution.plan.as_dict())
    submitted = read_json(output)
    if Plan.read(solver.graph, submitted) != solution.plan:
        raise AssertionError('导出的方案与选中的方案不一致')
    result = solver.settings.evaluate(solver.graph.raw, submitted)
    if result != solution.result:
        raise AssertionError('候选精评与原版第三问评测不一致')
    without = solver.settings.without_cache(solver.graph.raw, submitted)
    if result['data_movement_bytes'] != without['data_movement_bytes']:
        raise AssertionError('固定方案的两种配置产生了不同 COPY 字节')
    return result, without, dict(plan_sha256=digest(output),
                                 verification_seconds=time.perf_counter() - started, verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--cores', type=int, required=True, choices=range(1, 6))
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--budget', type=float, default=45, help='每种核数的搜索秒数；不含最终核验')
    parser.add_argument('--seeds', type=int, default=18)
    parser.add_argument('--local', type=int, default=32)
    parser.add_argument('--cache', type=int, default=16)
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()
    config = args.config or args.graph.parent / 'config.txt'
    if args.output.resolve() in (args.graph.resolve(), config.resolve()):
        parser.error('输出方案不能覆盖原始计算图或配置')
    solver = Solver(read_json(args.graph), CacheSettings.read(config))
    reports = []
    for n in range(1, args.cores + 1):
        solution, report = solver.solve(n, args.budget, args.seeds, args.local, args.cache, args.rounds)
        reports.append(report)
    result, without, verified = save_verified(solver, solution, args.output)
    write_json(args.output.with_name(args.output.stem + '_metrics.json'),
               dict(l2=compact(result), no_l2=compact(without), search=reports,
                    input_sha256=digest(args.graph), config_sha256=digest(config), **verified))
    print(f'T3={result["makespan"]}; T2(same plan)={without["makespan"]}; '
          f'cache speedup={without["makespan"] / max(1, result["makespan"]):.6f}; verified=True')


if __name__ == '__main__':
    main()

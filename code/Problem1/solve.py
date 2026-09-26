"""问题一场景 A 的切图、分核与调度搜索。"""

from dataclasses import dataclass
import time

from tools.graph import Graph
from Problem1.execution import Engine
from Problem1.partition import candidates, canonical, describe, feedback, refinements, bottleneck_bands
from Problem1.scheduling import list_schedule, neighbors, lower_bound, ranked_neighbors


@dataclass
class Solution:
    compiled: object
    schedules: list
    result: dict
    origin: str

    @property
    def score(self):
        return self.result['makespan'], self.result['data_movement_bytes']['added_copy_bytes']

    @property
    def plan(self):
        return self.compiled.partition.plan(self.schedules)


class Solver:
    """一个算例一个实例；顺序求解多个核数时复用图、子图展开和已有方案。"""
    def __init__(self, raw, settings):
        self.graph = Graph(raw)
        self.settings = settings
        self.engine = Engine(self.graph, settings)
        self.known = {}
        self.optimized = {}
        self.whole = None

    def singlecore(self):
        if self.whole is None:
            part = describe(self.graph, canonical([self.graph.order]), self.settings, 'whole')
            compiled = self.engine.prepare(part)
            orders = [part.order]
            self.whole = Solution(compiled, orders,
                                  self.engine.simulate(compiled, orders), 'whole')
        return self.whole

    def solve(self, cores, budget=180.0, max_candidates=3, moves=8):
        if not 1 <= cores <= 5:
            raise ValueError('赛题核数范围为 1～5')
        if budget <= 0 or max_candidates < 0 or moves < 0:
            raise ValueError('budget 必须为正，候选数与迁移次数必须非负')
        started = time.perf_counter()
        whole = self.singlecore()
        if cores == 1:
            report = dict(version='3.0', cores=1, selected='whole',
                          elapsed_seconds=time.perf_counter() - started,
                          baseline_makespan=whole.score[0], final_makespan=whole.score[0],
                          improvement_over_baseline=1.0, features=self.graph.features())
            return whole, whole, report
        deadline = started + budget
        before_preparations = self.engine.preparations
        before_builds, before_hits = self.engine.kernel_builds, self.engine.kernel_hits
        log, rejected = [], []
        best = None
        evaluated = set()
        duplicate_simulations = 0

        def consider(compiled, orders, label, required=False):
            nonlocal best, duplicate_simulations
            key = compiled.partition.groups, tuple(map(tuple, orders))
            if key in evaluated and not required:
                duplicate_simulations += 1
                return None
            evaluated.add(key)
            begin = time.perf_counter()
            result = self.engine.simulate(compiled, orders)
            solution = Solution(compiled, [list(o) for o in orders], result, label)
            improved = best is None or solution.score < best.score
            log.append(dict(candidate=label, tasks=len(compiled.partition.groups),
                            makespan=solution.score[0], added_copy_bytes=solution.score[1],
                            simulation_seconds=time.perf_counter() - begin, improved=improved))
            if improved:
                best = solution
            return solution

        consider(whole.compiled, whole.schedules + [[] for _ in range(cores - 1)], 'whole')
        for k in sorted(self.known):
            if k < cores:
                old = self.known[k]
                consider(old.compiled, old.schedules + [[] for _ in range(cores - k)],
                         f'carry_{k}_cores')

        pool, seen, baseline_entry = [], {}, None
        for name, groups in candidates(self.graph, cores, self.settings):
            if groups in seen:
                if name == 'chains_2':
                    baseline_entry = seen[groups]
                continue
            try:
                part = describe(self.graph, groups, self.settings, name)
            except ValueError as error:
                rejected.append(dict(candidate=name, reason=str(error)))
                continue
            orders, estimate = list_schedule(part, cores, self.settings)
            entry = (estimate, part.copy_bytes, name, part, orders)
            seen[groups] = entry
            pool.append(entry)
            if name == 'chains_2':
                baseline_entry = entry
        # 可复现的基础算法：拓扑链簇按周期等分 2N 组，再进行列表调度。
        baseline_entry = baseline_entry or pool[0]
        _, _, name, part, orders = baseline_entry
        baseline_compiled = self.engine.prepare(part)
        baseline = consider(baseline_compiled, orders, 'baseline_chain_list', required=True)
        exact_orders, _ = list_schedule(part, cores, self.settings, baseline_compiled.durations)
        consider(baseline_compiled, exact_orders, name + '/profiled')
        prepared = {whole.compiled.partition.groups, part.groups}

        def try_partition(part):
            if time.perf_counter() >= deadline:
                return
            if lower_bound(part, cores) > best.score[0]:
                rejected.append(dict(candidate=part.name, reason='时间下界已经超过当前最好结果'))
                return
            try:
                compiled = self.engine.prepare(part)
                prepared.add(part.groups)
                scheduled = set()
                for priority in ('critical', 'longest'):
                    orders, _ = list_schedule(part, cores, self.settings,
                                              compiled.durations, priority)
                    key = tuple(map(tuple, orders))
                    if key not in scheduled:
                        scheduled.add(key)
                        consider(compiled, orders, part.name + '/' + priority)
            except (ValueError, RuntimeError) as error:
                rejected.append(dict(candidate=part.name, reason=str(error)))

        count = 0
        for _, _, _, part, _ in sorted(pool, key=lambda row: row[:3]):
            if part.groups in prepared:
                continue
            if count >= max_candidates or time.perf_counter() >= deadline:
                break
            try_partition(part)
            count += 1
        def refine_best():
            for name, groups in feedback(self.graph, best.compiled.partition,
                                         best.schedules, best.result):
                if groups in prepared or time.perf_counter() >= deadline:
                    continue
                try:
                    try_partition(describe(self.graph, groups, self.settings, name))
                except ValueError as error:
                    rejected.append(dict(candidate=name, reason=str(error)))

        refine_best()
        for orders in neighbors(best.compiled.partition, best.schedules, best.result, moves):
            if time.perf_counter() >= deadline:
                break
            consider(best.compiled, orders, 'schedule_move')

        coarse_makespan = best.score[0]
        # 第一轮的最好结果始终保留；第二轮给不同结构的候选各一次机会。
        pool, families = [], set()
        for name, groups in refinements(self.graph, cores, self.settings):
            if time.perf_counter() >= deadline:
                break
            if groups in seen:
                continue
            seen[groups] = None
            try:
                part = describe(self.graph, groups, self.settings, name)
                orders, estimate = list_schedule(part, cores, self.settings)
                pool.append((estimate, part.copy_bytes, name, part))
            except ValueError as error:
                rejected.append(dict(candidate=name, reason=str(error)))
        count = 0
        for _, _, name, part in sorted(pool, key=lambda row: row[:3]):
            family = name.split('_')[0]
            if family in families:
                continue
            if count >= max_candidates or time.perf_counter() >= deadline:
                break
            families.add(family)
            try_partition(part)
            count += 1
        # 拆分/合并只在确实改善后继续，最多两轮。
        for _ in range(2):
            previous_score = best.score
            refine_best()
            if best.score == previous_score or time.perf_counter() >= deadline:
                break
        self.known[cores] = best
        backbone_makespan = best.score[0]
        features = self.graph.features()
        work = features['pipe_work']
        global_bound = max(max(self.graph.rank.values(), default=0),
                           max((value / cores for value in work.values()), default=0))
        # 保留原搜索路径及其跨核数传递；增强结果只在此后的阶段参与比较。
        if max_candidates and best.score[0] > 1.08 * global_bound:
            pool = []
            for name, groups in bottleneck_bands(
                    self.graph, best.compiled.partition, best.result, cores, self.settings):
                if time.perf_counter() >= deadline:
                    break
                if groups in seen or groups in prepared:
                    continue
                seen[groups] = None
                try:
                    part = describe(self.graph, groups, self.settings, name)
                    if lower_bound(part, cores) <= best.score[0]:
                        _, estimate = list_schedule(part, cores, self.settings)
                        pool.append((estimate, part.copy_bytes, name, part))
                except ValueError:
                    continue
            for _, _, _, part in sorted(pool, key=lambda row: row[:3])[:min(2, max_candidates)]:
                try_partition(part)
        for k, old in sorted(self.optimized.items()):
            if k < cores and old.score < best.score:
                consider(old.compiled, old.schedules + [[] for _ in range(cores - k)],
                         f'polished_carry_{k}')
        # 细化后的最终切图也得到调度优化；仅在实际改善后开始下一轮。
        for _ in range(2):
            if time.perf_counter() >= deadline or len(best.compiled.partition.groups) > 256:
                break
            previous_score = best.score
            compiled = best.compiled
            for orders in ranked_neighbors(compiled, best.schedules, best.result, self.settings, moves):
                if time.perf_counter() >= deadline:
                    break
                consider(compiled, orders, 'final_schedule_move')
            if best.score == previous_score:
                break
        self.optimized[cores] = best
        report = dict(
            version='3.0', cores=cores, selected=best.origin,
            elapsed_seconds=time.perf_counter() - started, search_budget_seconds=budget,
            budget_note='在候选之间检查预算；已经启动的核内展开不被中断。最终原评测另计时。',
            max_candidates=max_candidates, moves=moves,
            preparations=self.engine.preparations - before_preparations,
            kernel_builds=self.engine.kernel_builds - before_builds,
            kernel_cache_hits=self.engine.kernel_hits - before_hits,
            duplicate_simulations_skipped=duplicate_simulations,
            simulations=len(log), candidates=log, rejected=rejected,
            baseline_makespan=baseline.score[0], final_makespan=best.score[0],
            coarse_makespan=coarse_makespan,
            backbone_makespan=backbone_makespan,
            improvement_over_baseline=baseline.score[0] / best.score[0] if best.score[0] else 1.0,
            lower_bound_cycles=global_bound,
            gap_to_lower_bound=best.score[0] / global_bound - 1 if global_bound else None,
            features=features)
        return best, baseline, report


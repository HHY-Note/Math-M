"""场景 B 独立求解：原图构造初始解，按官方结果改进排序和分核。"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import math
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.graph import Graph
from tools.official import ROOT, SceneBSettings, read_json, write_json
from tools.experiment import official_summary
from tools.preview import evaluate as preview
from Problem2.partition import CostModel, Plan, check_plan
from Problem2.scheduling import move_candidates, order_candidates
from evaluation_validation import EvaluationValidationError
from multicore_cut_evaluate_problem_2 import SceneBEvaluationError
from schedule_step2 import Step2SchedulingError
from schedule_step3 import Step3SchedulingError
from stub_multicore_cut_and_schedule import MulticoreCutError

EVALUATION_ERRORS = (ValueError, EvaluationValidationError, SceneBEvaluationError,
                     Step2SchedulingError, Step3SchedulingError, MulticoreCutError)


@dataclass
class Solution:
    plan: Plan
    result: dict
    origin: str

    @property
    def score(self):
        return self.result['makespan'], self.result['data_movement_bytes']['added_copy_bytes']


class Solver:
    """输入只含原图与配置；全部成绩由当前进程计算，保留已找到的最好解。"""
    def __init__(self, raw, settings):
        self.graph = Graph(raw)
        self.settings = settings
        self.model = CostModel(self.graph, settings)
        self.known = {}

    @property
    def single(self):
        if 1 not in self.known:
            plan = Plan.build([[self.graph.order]])
            # 整图一个 Task 时 A/B 无跨核、无 Task 间等待，执行图与时间一致。
            # 使用官方 B 入口省掉 A 入口逐事件扫描完整 Task 的额外开销。
            result = self.settings.evaluate(self.graph.raw, plan.as_dict())
            self.known[1] = Solution(plan, result, 'whole')
        return official_summary(self.known[1].result)

    def solve(self, cores, budget=45.0, candidates=24, orders=16, moves=8):
        if type(cores) is not int or cores not in range(1, 6):
            raise ValueError('核心数必须为 1～5 的整数')
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError('budget 必须为正的有限数')
        if any(type(v) is not int or v < 0 for v in (candidates, orders, moves)):
            raise ValueError('候选数量必须为非负整数')
        single = self.single
        budget = min(budget, max(18, 10 + 0.0015 * len(self.graph.ops)))
        started = time.perf_counter()
        old = min((s for n, s in self.known.items() if n <= cores), key=lambda s: s.score)
        best = Solution(old.plan.padded(cores), old.result, f'carry_{len(old.plan.blocks)}')
        history, attempted, frontier = [], {best.plan}, [best]
        lower_bound = self.model.lower_bound(cores)
        # 相对全局最优时间的可证差距至多 0.5% 时，节省剩余搜索时间。
        def near_optimal():
            return best.score[0] * 1000 <= lower_bound * 1005

        def evaluate(plan, name, stage):
            nonlocal best
            if plan is None or plan in attempted:
                return False
            attempted.add(plan)
            begin = time.perf_counter()
            called = False
            try:
                check_plan(self.graph, plan)
                if plan not in self.model.bounds:
                    self.model.estimate(plan)
                bound = self.model.bounds[plan]
                if bound > best.score[0]:
                    history.append(dict(stage=stage, candidate=name, status='pruned', preview_call=False,
                                        lower_bound=bound, seconds=time.perf_counter() - begin))
                    return False
                called = True
                result = preview(self.graph.raw, plan.as_dict(), self.settings)
                solution = Solution(plan, result, name)
                if stage in ('partition', 'refine'):
                    diverse, assignments = [], set()
                    for value in sorted(frontier + [solution], key=lambda s: s.score):
                        key = tuple(sorted(value.plan.assignment().items()))
                        if key not in assignments:
                            assignments.add(key)
                            diverse.append(value)
                    coarse = min((s for s in diverse if s.score[0] <= diverse[0].score[0] * 1.5),
                                 key=lambda s: (sum(map(len, s.plan.blocks)), s.score))
                    light = min((s for s in diverse if s.score[0] <= diverse[0].score[0] * 1.3),
                                key=lambda s: (s.score[1], s.score[0]))
                    frontier[:] = diverse[:2]
                    for value in (light, coarse):
                        if value not in frontier:
                            frontier.append(value)
                improved = solution.score < best.score
                history.append(dict(stage=stage, candidate=name, status='ok', preview_call=True,
                                    makespan=solution.score[0], added_bytes=solution.score[1],
                                    subgraphs=sum(map(len, plan.blocks)), improved=improved,
                                    seconds=time.perf_counter() - begin))
                if improved:
                    best = solution
                return improved
            except EVALUATION_ERRORS as error:
                history.append(dict(stage=stage, candidate=name, status='rejected', preview_call=called,
                                    error=f'{type(error).__name__}: {str(error)[:500]}',
                                    seconds=time.perf_counter() - begin))
                return False

        if cores > 1 and candidates and not near_optimal():
            pool, seen = [], set(attempted)
            for name, plan in self.model.candidates(cores):
                if plan is None or plan in seen:
                    continue
                seen.add(plan)
                pool.append((self.model.estimate(plan), name, plan))
            pool.sort(key=lambda item: (item[0], item[1]))
            # 先比较不同结构，避免代理误差使同类候选占满预算。
            chosen, families = [], set()
            for item in pool:
                family = item[1].split('_')[0]
                if family not in families:
                    chosen.append(item)
                    families.add(family)
            chosen.extend(item for item in pool if item not in chosen)
            # 完整独立分支不产生核间通信，在大图限时搜索中优先保留这个候选。
            depth = max(self.graph.depth.values(), default=0)
            large = len(self.graph.ops) > 4096
            regular_deep = large and depth > 32 and len(self.graph.ops) >= depth * len(self.graph.components) / 2
            primary = ('fork_reduce' if regular_deep and len(self.graph.components) >= cores else
                       'components_1' if len(self.graph.components) >= cores else
                       'window_4' if depth * depth > 16 * len(self.graph.ops) else
                       'fork_reduce' if depth > 128 else 'levels_2')
            coarse_first = (primary == 'components_1' and
                            (depth > 32 or len(self.graph.ops) > 2 * depth * len(self.graph.components)))
            chosen.sort(key=lambda item: (item[1] != primary,
                        not (coarse_first and item[1] == 'coarsen')))
            for _, name, plan in chosen[:candidates]:
                if near_optimal() or (history and time.perf_counter() - started >= budget * 0.5):
                    break
                evaluate(plan, name, 'partition')
            for round_id in range(2):
                if near_optimal() or time.perf_counter() - started >= budget * 0.6:
                    break
                seed = (min(frontier, key=lambda s: (sum(map(len, s.plan.blocks)), s.score))
                        if round_id == 0 and not large else best)
                pool = [(self.model.estimate(plan), name, plan)
                        for name, plan in self.model.refinements(seed) if plan not in attempted]
                pool.sort(key=lambda item: (item[0], item[1]))
                if large and pool:
                    close = sorted((item for item in pool[1:] if item[0][0] <= pool[0][0][0] * 1.01),
                                   key=lambda item: (item[0][1], int(item[1].rsplit('_', 1)[1])))
                    pool = pool[:1] + close[:2] + [item for item in pool[1:] if item not in close[:2]]
                for _, name, plan in pool[:3]:
                    if near_optimal() or time.perf_counter() - started >= budget * 0.65:
                        break
                    evaluate(plan, name, 'refine')
        initial = best.score

        if cores > 1 and orders:
            count, rounds = 0, 0
            while count < orders and not near_optimal() and time.perf_counter() - started < budget * 0.9:
                incumbent = best
                bases, seen = [best], {tuple(sorted(best.plan.assignment().items()))}
                for candidate in sorted(frontier, key=lambda s: s.score):
                    key = tuple(sorted(candidate.plan.assignment().items()))
                    if key not in seen and candidate.score[0] <= best.score[0] * 1.3:
                        seen.add(key)
                        bases.append(candidate)
                choices = [list(order_candidates(self.model, s.plan)) for s in bases]
                for base, row in zip(bases, choices):
                    depth = max(self.graph.depth.values(), default=0)
                    first = ('split_release' if base.result['cross_core_transfers'] else
                             'critical_runs' if depth > 32 else
                             'release_runs' if base.result['data_movement_bytes']['spill_added_copy_bytes']
                             else 'split_release')
                    priority = ('merge_neighbors' if first == 'critical_runs'
                                and sum(map(len, base.plan.blocks)) > cores
                                and not base.result['cross_core_transfers'] else first)
                    row.sort(key=lambda item: item[0] != priority)
                options = [(f'{row[i][0]}@{base.origin}', row[i][1])
                           for i in range(max(map(len, choices))) for base, row in zip(bases, choices)
                           if i < len(row)]
                for name, plan in options:
                    if count >= orders or near_optimal() or time.perf_counter() - started >= budget * 0.9:
                        break
                    if plan is not None and plan not in attempted:
                        evaluate(plan, name, 'order')
                        count += 1
                rounds += 1
                if best.score >= incumbent.score or rounds >= 2:
                    break
        ordered = best.score

        if cores > 1 and moves and not near_optimal() and time.perf_counter() - started < budget:
            for name, plan in move_candidates(self.model, best, limit=moves):
                if near_optimal() or time.perf_counter() - started >= budget:
                    break
                evaluate(plan, name, 'move')
        search_seconds = time.perf_counter() - started
        # 对最终方案重新调用原版评测，逐项比较完整结果，包括每个算子的时间线。
        if best.result['num_cores'] != cores:
            best.result = preview(self.graph.raw, best.plan.as_dict(), self.settings)
        begin = time.perf_counter()
        result = self.settings.evaluate(self.graph.raw, best.plan.as_dict())
        if result != best.result:
            raise AssertionError('候选评估与原版官方结果不一致，不输出方案')
        best.result = result
        verification_seconds = time.perf_counter() - begin
        self.known[cores] = best
        report = dict(version='2.3', independent=True, cores=cores, selected=best.origin,
                      singlecore_makespan=single['makespan'], carry_makespan=old.score[0],
                      initial_makespan=initial[0], ordered_makespan=ordered[0], final_makespan=best.score[0],
                      elapsed_seconds=time.perf_counter() - started, budget=budget,
                      search_seconds=search_seconds, verification_seconds=verification_seconds,
                      budget_exhausted=search_seconds >= budget,
                      official_calls=1, preview_calls=sum(r['preview_call'] for r in history),
                      rejected=sum(r['status'] == 'rejected' for r in history),
                      pruned=sum(r['status'] == 'pruned' for r in history),
                      official_verified=True, history=history)
        report.update(global_lower_bound=lower_bound,
                      gap_upper_bound=best.score[0] / lower_bound - 1 if lower_bound else 0,
                      near_optimal=near_optimal())
        return best, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path)
    parser.add_argument('-n', '--num-cores', type=int, choices=range(1, 6), required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('-o', '--output', type=Path)
    parser.add_argument('--budget', type=float, default=45)
    parser.add_argument('--candidates', type=int, default=24)
    parser.add_argument('--orders', type=int, default=16)
    parser.add_argument('--moves', type=int, default=8)
    args = parser.parse_args()
    solver = Solver(read_json(args.graph), SceneBSettings.read(args.config or args.graph.parent / 'config.txt'))
    for n in ([1] if args.num_cores == 1 else range(2, args.num_cores + 1)):
        best, report = solver.solve(n, args.budget, args.candidates, args.orders, args.moves)
    output = args.output or ROOT / 'results/Problem2/single' / (args.graph.stem + '_multicore_res.json')
    write_json(output, best.plan.as_dict())
    write_json(output.with_name(output.stem + '_search.json'), report)
    write_json(output.with_name(output.stem + '_official.json'), official_summary(best.result))
    print(f'{args.graph.stem}: N={args.num_cores}; T={best.score[0]}; '
          f'added_bytes={best.score[1]}; independent=True; official_verified=True')


if __name__ == '__main__':
    main()

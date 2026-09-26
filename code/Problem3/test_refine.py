"""精修的保底约束、时间线候选合法性与官方一致性回归。

运行：python -m unittest discover -s code/Problem3 -p test_refine.py -v
"""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Problem3.search import timeline_neighbors
from Problem3.solve import Solution, Solver
from tools.graph import Graph
from tools.official import CacheSettings
from tools.plan import Plan, check_plan


def independent_graph(count=4):
    return Graph(dict(
        ops=[dict(id=i, op='COMPUTE', pipe='PIPE_M', cycles=5)
             for i in range(1, count + 1)],
        tensors=[], edges=[]))


def score_result(makespan, added):
    return dict(makespan=makespan,
                data_movement_bytes=dict(added_copy_bytes=added))


class RefineGuardTests(unittest.TestCase):
    def setUp(self):
        self.graph = independent_graph()
        self.anchor = Plan.build([[[1, 2, 3, 4]]])
        self.a = Plan.build([[[1, 2], [3, 4]]])
        self.b = Plan.build([[[1], [2], [3, 4]]])
        self.c = Plan.build([[[1], [2], [3], [4]]])

    def run_refine(self, rows, *, base=(100, 20, 200), budget=100, limit=64):
        """使用互不依赖的合法方案，单独检验真实 refine 接受逻辑。"""
        original = Solution(self.anchor, score_result(base[0], base[1]), 'original')
        solver = Solver.__new__(Solver)
        solver.graph = self.graph
        solver.model = SimpleNamespace(estimate=Mock(return_value=(0, 0)))
        solver.known = {1: original}
        values = {self.anchor: base}
        values.update((plan, numbers) for _, plan, numbers in rows if plan is not None)

        def metrics(submitted):
            return values[Plan.read(self.graph, submitted)]

        solver.evaluate = Mock(side_effect=lambda submitted:
                               score_result(*metrics(submitted)[:2]))
        solver.settings = SimpleNamespace(without_cache=Mock(
            side_effect=lambda raw, submitted: dict(makespan=metrics(submitted)[2])))
        proposals = [(name, plan) for name, plan, _ in rows]
        before = deepcopy(original.result)
        known = dict(solver.known)
        with patch('Problem3.solve.timeline_neighbors', return_value=iter(proposals)):
            result, report = solver.refine(original, budget, limit)
        self.assertEqual(original.result, before)
        self.assertEqual(solver.known, known)
        self.assertIs(solver.known[1], original)
        return result, report, solver, original

    def test_zero_budget_or_limit_retains_original(self):
        for budget, limit in ((0, 64), (100, 0)):
            with self.subTest(budget=budget, limit=limit):
                result, report, solver, original = self.run_refine(
                    [('better', self.a, (80, 10, 180))], budget=budget, limit=limit)
                self.assertIs(result, original)
                self.assertEqual(report['candidates'], 0)
                solver.evaluate.assert_not_called()

    def test_each_protected_metric_blocks_regression(self):
        for label, numbers in (
                ('l2_time', (101, 19, 200)),
                ('added_bytes', (80, 21, 180)),
                ('no_l2_time', (90, 20, 201)),
                ('cache_ratio', (90, 20, 179))):
            with self.subTest(metric=label):
                result, _, _, original = self.run_refine([(label, self.a, numbers)])
                self.assertIs(result, original)

    def test_better_time_and_equal_ratio_are_accepted(self):
        result, report, _, _ = self.run_refine([('better', self.a, (90, 20, 180))])
        self.assertEqual(result.plan, self.a)
        self.assertEqual(report['final_makespan'], 90)
        self.assertEqual(report['final_no_l2_makespan'], 180)

    def test_equal_time_with_less_movement_is_accepted(self):
        result, _, _, _ = self.run_refine([('less_copy', self.a, (100, 19, 200))])
        self.assertEqual(result.plan, self.a)

    def test_fixed_anchor_does_not_ratchet_ratio_or_no_l2_threshold(self):
        # A 的 2.50 倍收益不能把门槛从原方案的 2.00 倍提高到 2.50 倍。
        rows = [('ratio_first', self.a, (80, 20, 200)),
                ('faster_next', self.b, (70, 20, 150)),
                ('fastest', self.c, (60, 20, 160))]
        result, report, _, _ = self.run_refine(rows)
        self.assertEqual(result.plan, self.c)
        self.assertEqual(report['final_makespan'], 60)
        self.assertEqual(sum(bool(row.get('improved')) for row in report['history']), 3)

    def test_ratio_comparison_preserves_integer_precision(self):
        n = 10 ** 18
        # 浮点除法会把两个比值都舍入成 1.0；精确乘积仍必须拒绝下降。
        result, _, _, original = self.run_refine(
            [('tiny_ratio_loss', self.a, (n - 1, 20, n - 1))],
            base=(n, 20, n + 1))
        self.assertIs(result, original)
        result, _, _, _ = self.run_refine(
            [('tiny_ratio_gain', self.a, (n - 1, 20, n))],
            base=(n, 20, n + 1))
        self.assertEqual(result.plan, self.a)

    def test_invalid_duplicate_and_failed_candidates_preserve_incumbent(self):
        invalid = Plan.build([[[1, 2, 3]]])
        rows = [('none', None, (0, 0, 0)), ('duplicate', self.anchor, (100, 20, 200)),
                ('missing_node', invalid, (1, 0, 1)),
                ('valid', self.a, (90, 20, 180)), ('duplicate_again', self.a, (90, 20, 180))]
        result, report, solver, _ = self.run_refine(rows)
        self.assertEqual(result.plan, self.a)
        self.assertEqual(solver.evaluate.call_count, 1)
        self.assertEqual(report['candidates'], 1)
        self.assertTrue(any(row['status'] == 'rejected' for row in report['history']))

    def test_candidate_budget_counts_evaluations(self):
        result, report, solver, _ = self.run_refine(
            [('first', self.a, (90, 20, 180)), ('second', self.b, (80, 20, 170))], limit=1)
        self.assertEqual(result.plan, self.a)
        self.assertEqual(report['candidates'], 1)
        self.assertEqual(solver.evaluate.call_count, 1)

    def test_invalid_anchor_is_rejected_before_evaluation(self):
        solver = Solver.__new__(Solver)
        solver.graph = self.graph
        solver.settings = SimpleNamespace(without_cache=Mock())
        with self.assertRaises(ValueError):
            solver.refine(Solution(Plan.build([[[1]]]), score_result(1, 0), 'invalid'))
        solver.settings.without_cache.assert_not_called()

    def test_invalid_limits_fail_explicitly(self):
        solver = Solver.__new__(Solver)
        original = Solution(self.anchor, score_result(100, 20), 'original')
        for budget, limit in ((-1, 1), (float('inf'), 1), (float('nan'), 1),
                              (1, -1), (1, 1.5), (1, True)):
            with self.subTest(budget=budget, limit=limit), self.assertRaises(ValueError):
                solver.refine(original, budget, limit)


class TimelineNeighborTests(unittest.TestCase):
    @staticmethod
    def fixture():
        # 每核交错发射，故 width=4 仍产生单节点段；1 -> 3 -> 7 限制核0移动。
        graph = independent_graph(10)
        graph.raw['tensors'] = [dict(id=100, pos='UB', size=8, logical_tid=900),
                                dict(id=101, pos='UB', size=8, logical_tid=900)]
        graph.raw['edges'] = [dict(source=100, target=3), dict(source=101, target=8),
                             dict(source=1, target=3), dict(source=3, target=7)]
        graph = Graph(graph.raw)
        plan = Plan.build([[[1, 3, 5, 7, 9]], [[2, 4, 6, 8, 10]], []])
        timeline = [dict(core_id=c, ops=[dict(op_id=v, start=10*v, end=10*v+5)
                    for group in core for v in group]) for c, core in enumerate(plan.blocks)]
        events = [dict(event='miss', tensor_id=900, size_bytes=8, core_id=c)
                  for c in (0, 1)]
        solution = Solution(plan, dict(per_core_timeline=timeline, cache_events=events), 'fixture')
        model = SimpleNamespace(graph=graph, settings=SimpleNamespace(cache_capacity=64))
        return graph, model, solution

    def test_all_candidates_preserve_assignment_coverage_and_legal_order(self):
        graph, model, solution = self.fixture()
        before = deepcopy(solution.result)
        proposals = list(timeline_neighbors(model, solution))
        self.assertTrue(any(name.startswith('phase_') for name, _ in proposals))
        for name, plan in proposals:
            with self.subTest(candidate=name):
                self.assertIsNotNone(plan)
                check_plan(graph, plan)
                self.assertEqual(plan.assignment(), solution.plan.assignment())
                self.assertEqual(len(plan.blocks), 3)
        self.assertEqual(solution.result, before)

    def test_logical_tensor_aliases_find_both_physical_consumers(self):
        _, model, solution = self.fixture()
        names = [name for name, _ in timeline_neighbors(model, solution)]
        self.assertTrue(any(name.startswith('phase_900_0_') for name in names))
        self.assertTrue(any(name.startswith('phase_900_1_') for name in names))
        self.assertFalse(any(name.startswith(('phase_100_', 'phase_101_')) for name in names))

    def test_phase_never_crosses_same_core_dependency(self):
        _, model, solution = self.fixture()
        names = {name for name, _ in timeline_neighbors(model, solution)}
        self.assertIn('phase_900_0_1', names)
        self.assertNotIn('phase_900_0_-1', names)  # 不能越过前驱1。
        self.assertNotIn('phase_900_0_2', names)   # 不能越过后继7。

    def test_unique_and_oversized_misses_do_not_create_cache_phases(self):
        for events in ([dict(event='miss', tensor_id=900, size_bytes=8, core_id=0)],
                       [dict(event='miss', tensor_id=900, size_bytes=65, core_id=c)
                        for c in (0, 1)]):
            with self.subTest(events=events):
                _, model, solution = self.fixture()
                solution.result['cache_events'] = events
                self.assertFalse(any(name.startswith('phase_')
                                     for name, _ in timeline_neighbors(model, solution)))

    def test_official_evaluation_accepts_every_small_candidate(self):
        graph, model, solution = self.fixture()
        settings = CacheSettings(dict(L1=524288, UB=131072), 60, 500, 1048576, 250)
        for name, plan in timeline_neighbors(model, solution):
            with self.subTest(candidate=name):
                result = settings.evaluate(graph.raw, plan.as_dict())
                without = settings.without_cache(graph.raw, plan.as_dict())
                self.assertEqual(result['data_movement_bytes'], without['data_movement_bytes'])
                self.assertEqual(result['num_cores'], 3)


if __name__ == '__main__':
    unittest.main()

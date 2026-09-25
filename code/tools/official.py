"""只读调用原题包；不复制或修改评测算法、配置和数据。"""

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "题目" / "code"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))

from contest_io import _read_json as read_json
from evaluation_validation import read_evaluation_config, validate_graph
from multicore_cut_evaluate_problem_1 import (
    _copy_traffic_bytes, evaluate_scene_a, read_scene_a_config,
)
from multicore_cut_evaluate_problem_2 import evaluate_scene_b, read_scene_b_config
from multicore_cut_evaluate_problem_3 import evaluate_problem_3, read_cache_config
from singlecore_evaluate import evaluate_singlecore
from schedule_step1 import step1_schedule
from schedule_step2 import step2_spill_insertion, _build_extended_graph
from schedule_step3 import PIPES, _op_duration, _uses_ddr_bandwidth, prepare_step3_execution
from stub_multicore_cut_and_schedule import (
    _build_op_adjacency, _contract_excluded_copy_nodes, derive_multicore_plan,
)


@dataclass(frozen=True)
class Settings:
    capacity: dict
    bandwidth: int
    same_wait: int
    cross_wait: int

    @classmethod
    def read(cls, path):
        values = read_evaluation_config(str(path))
        waits = read_scene_a_config(str(path))
        return cls(values['capacity'], values['bandwidth'],
                   waits['task_same_core_wait_cycles'],
                   waits['task_cross_core_wait_cycles'])

    def evaluate(self, graph, plan):
        return evaluate_scene_a(
            graph, plan, self.bandwidth, self.capacity,
            self.cross_wait, self.same_wait)


@dataclass(frozen=True)
class SceneBSettings:
    capacity: dict
    bandwidth: int
    copy_delay: int

    @classmethod
    def read(cls, path):
        values = read_evaluation_config(str(path))
        delay = read_scene_b_config(str(path))
        return cls(values['capacity'], values['bandwidth'],
                   delay['cross_core_copy_delay_cycles'])

    def evaluate(self, graph, plan):
        return evaluate_scene_b(graph, plan, self.bandwidth, self.capacity,
                                self.copy_delay)

    def singlecore(self, graph):
        return evaluate_singlecore(graph, self.bandwidth, self.capacity)


@dataclass(frozen=True)
class CacheSettings(SceneBSettings):
    cache_capacity: int
    cache_bandwidth: float

    @classmethod
    def read(cls, path):
        base = SceneBSettings.read(path)
        cache = read_cache_config(str(path))
        return cls(base.capacity, base.bandwidth, base.copy_delay,
                   cache['cache_capacity_bytes'], cache['cache_bandwidth_bytes_per_cycle'])

    def evaluate(self, graph, plan):
        return evaluate_problem_3(graph, plan, self.bandwidth, self.capacity, self.copy_delay,
                                  self.cache_capacity, self.cache_bandwidth)

    def without_cache(self, graph, plan):
        return evaluate_scene_b(graph, plan, self.bandwidth, self.capacity, self.copy_delay)


def write_json(path, value):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8')

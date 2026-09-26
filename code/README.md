# 通用神经网络处理器多核调度：代码说明

本目录按“通用工具 + 三个问题”组织。本文说明提交代码的职责、输入输出、运行方式和
复现实验；每题的建模与算法说明见对应的 `Problem1/README.md`、`Problem2/README.md`、
`Problem3/README.md`。

## 1. 目录结构

```text
code/
├── README.md
├── tools/
│   ├── graph.py
│   ├── plan.py
│   ├── official.py
│   ├── preview.py
│   ├── experiment.py
│   └── plot_results.py
├── Problem1/
│   ├── README.md
│   ├── solve.py
│   ├── partition.py
│   ├── scheduling.py
│   ├── execution.py
│   ├── benchmark.py
│   └── plot_results.py
├── Problem2/
│   ├── README.md
│   ├── solve.py
│   ├── partition.py
│   ├── scheduling.py
│   └── benchmark.py
└── Problem3/
    ├── README.md
    ├── model.py
    ├── search.py
    ├── diagnostics.py
    ├── solve.py
    ├── benchmark.py
    ├── sensitivity.py
    └── plot_results.py
```

三个问题均可从原图独立运行。问题三不读取问题一、二的方案或搜索状态；
`tools/` 只提供图处理、方案表示、官方接口适配、精评复用和实验输出。

## 2. 通用接口

| 文件 | 主要接口 | 职责 |
| --- | --- | --- |
| `tools/graph.py` | `Graph`、`topological()` | 校验原图，建立算子 DAG、张量关系、拓扑序、深度、连通分量和安全链 |
| `tools/plan.py` | `Plan`、`check_plan()`、`arrange()`、`runs()` | 表示子图/核心/顺序，检查覆盖、子图 DAG 和同核顺序 |
| `tools/official.py` | `Settings`、`SceneBSettings`、`CacheSettings` | 读取固定配置，只读调用题包官方评测器 |
| `tools/preview.py` | `Evaluator`、`evaluate()` | 复用相同核内准备或执行结果，加速候选精评；不改变官方语义 |
| `tools/experiment.py` | `digest()`、`write_csv()` | 文件散列、JSON 摘要和 CSV 输出 |
| `tools/plot_results.py` | `plot()` | 共用实验曲线绘制 |

`Graph` 同时保留算子依赖和张量生产者/消费者集合。节点 ID 可以稀疏、不连续。
`Plan` 的标准输出只包含 `node_to_subgraph` 和 `core_schedules` 两个字段；
所有评测器都再次检查输入、容量、依赖和全局执行图。

官方模块从题包提供的 `code` 目录只读导入，代码不复制、不修改官方评测算法和配置。
输入图默认从同目录 `config.txt` 读取配置；批量入口可通过 `--data` 和 `--config` 明确指定。
配置缺失、键错误、重复 JSON 键或非法方案都会报错，不使用算法内置参数静默替代。

## 3. 三个问题的实现职责

### 3.1 问题一

问题一对应场景 A，每个子图是一个独立 Task，跨子图通信由官方程序插入 COPY。

| 文件 | 职责 |
| --- | --- |
| `Problem1/solve.py` | 候选搜索、递增核数保底、官方核验和提交文件 |
| `Problem1/partition.py` | 分区表示、结构候选、瓶颈分层与边界代价 |
| `Problem1/scheduling.py` | 列表调度、时间下界、迁移邻域和候选排序 |
| `Problem1/execution.py` | 局部图、精确多核模拟、逐操作结果检查 |
| `Problem1/benchmark.py` | 批量用例、指标汇总、历史目录对照 |
| `Problem1/plot_results.py` | 平均加速比曲线 |

### 3.2 问题二

问题二对应场景 B，同一核心上的子图合为一个 Task，允许片上数据驻留并通过跨核同步通信。
主要算法、模型、输入输出格式和验证方法见 `Problem2/README.md`。

### 3.3 问题三

问题三在问题二执行模型上加入只读 FIFO L2 Cache。`Problem3/solve.py` 的基础
`Solver.solve()` 负责结构候选、下界筛选、官方精评和反馈搜索；
`Solver.refine()` 在固定方案上追加独立的时间线细分与 Cache 热点相位搜索。

追加阶段满足以下固定基准守卫：

$$
T_C\le T_C^0,\qquad A\le A^0,\qquad T_D\le T_D^0,
\qquad T_D T_C^0\ge T_D^0 T_C.
$$

其中上标 0 表示进入追加阶段时由当前官方重新评分的方案。
四项检查用整数交叉相乘完成，接受结果按 `(T_C,A)` 字典序比较。
追加预算不挤占基础搜索，不写入跨核数 `known` 保底状态。

| 文件 | 职责 |
| --- | --- |
| `Problem3/model.py` | 张量代价、Pipe 工作量、代理模型、全局和顺序安全下界、初始结构候选 |
| `Problem3/search.py` | 排序、细分、合并、重切、迁移、原 Cache 反馈及 `timeline_neighbors()` |
| `Problem3/solve.py` | 基础搜索、独立追加搜索、固定基准验收、提交与双配置核验 |
| `Problem3/diagnostics.py` | 真实 Cache 事件、冷 miss、填充后 miss、带宽和核心负载诊断 |
| `Problem3/benchmark.py` | 1-5 核批量运行、同源旧方案重评、逐配置比较和汇总 |
| `Problem3/plot_results.py` | 无 L2/L2 同方案时间曲线、Cache 加速比曲线 |
| `Problem3/sensitivity.py` | 固定方案的 L2 容量和带宽补充实验 |

## 4. 输入与输出格式

### 4.1 原始计算图

原图 JSON 顶层必须包含 `ops`、`tensors`、`edges` 三个数组。
节点 ID 在全图唯一，数值字段非负，边只能连接 Tensor 与 Op，完整图必须无环。

```json
{
  "tensors": [{"id": 1, "pos": "DDR", "size": 1024}],
  "ops": [{"id": 2, "op": "ADD", "pipe": "PIPE_V", "cycles": 20}],
  "edges": [{"source": 1, "target": 2}]
}
```

`COPY_IN`、`COPY_OUT` 是原图中的搬运操作，选手方案不为它们指定子图；
方案只覆盖其他操作。`pipe` 必须是题包规定的四条 Pipe 之一。

### 4.2 选手方案

顶层只能包含以下两个字段：

```json
{
  "node_to_subgraph": {"11": 0, "12": 1},
  "core_schedules": [[0], [1]]
}
```

映射必须覆盖全部非 COPY 操作且无重复。每个子图必须只出现在一个核心列表中，
外层列表长度等于指定核心数，允许空核心。官方评测器根据方案重建边界 COPY、
Spill/Reload、Pipe 顺序、跨核等待和 Cache 事件。

### 4.3 评测指标

- `makespan`：所有操作完成时刻的最大值，主要目标。
- `data_movement_bytes.added_copy_bytes`：相对于原图新增的 DDR 搬运字节。
- `cache_stats.hit_rate`：命中字节占可缓存访问字节的比例。
- `speedup_l2`、`speedup_no_l2`：相对于整图单核无 L2 基准的并行加速比。
- `cache_speedup`：同一方案的 `no_l2_makespan / makespan`。

Cache 命中仍计入官方 COPY 统计；命中率不替代 Makespan，也不作为唯一接受条件。

## 5. 运行依赖与固定配置

- Python 3.10 或更高版本。
- 求解、官方评测和批量实验只依赖 Python 标准库。
- SVG 绘图需要 ReportLab；PNG 导出另需 pypdfium2。

固定配置由题包 `config.txt` 提供。问题三读取 L1、UB、DDR、跨核等待以及
L2 容量和带宽，代码不为缺少配置提供默认成绩。

## 6. 运行方法

以下命令在项目根目录执行。`<DATA_DIR>` 是含计算图和 `config.txt` 的数据目录，
`<OUTPUT_DIR>` 必须是不存在或空目录。

### 6.1 单用例

```powershell
python -B -X utf8 code/Problem1/solve.py <DATA_DIR>/case_001.json -n 4 -o results/Problem1/case_001_multicore_res.json
python -B -X utf8 code/Problem2/solve.py <DATA_DIR>/case_001.json -n 4 -o results/Problem2/case_001_multicore_res.json
python -B -X utf8 code/Problem3/solve.py <DATA_DIR>/case_044.json --cores 5 -o results/Problem3/case_044_multicore_res.json
```

问题三入口默认先依次求解 1 至目标核数，再对目标方案执行独立追加优化。
单用例可用 `--refine-budget 0` 关闭追加阶段；`--refine-candidates` 控制追加 L2 精评次数。
使用 `--initial-plan FILE` 时，程序先用当前官方评测器重新评该方案，再只做固定基准追加优化。

### 6.2 问题三批量实验

```powershell
python -B -X utf8 code/Problem3/benchmark.py --data <DATA_DIR> --cases 1 2 4 8 10 14 16 20 29 37 44 46 58 69 71 72 74 80 90 93 --workers 3 --output <OUTPUT_DIR>
```

默认用例集由 `Problem3/benchmark.py` 的 `DEFAULT_CASES` 给出，明确指定 `--cases` 可复现同一集合。
每个用例运行 1-5 核并保存 `baseline.json`、逐核方案、`official.json`、`diagnostics.json`、
`search.json`、`refinement.json`、`metrics.csv` 和 `summary.json`。

从历史第三问结果继续追加优化：

```powershell
python -B -X utf8 code/Problem3/benchmark.py --data <DATA_DIR> --cases 1 44 90 --refine-from <BASELINE_DIR> --refine-budget 15 --refine-candidates 64 --compare <BASELINE_DIR> --output <OUTPUT_DIR>
```

`--refine-from` 会检查输入图和配置散列，并逐份方案执行当前官方双配置复评。
旧 `reference_sha256` 为空时，只有全部方案复评的 `compact()` 结果精确匹配才允许继续比较；
非空且不一致的官方源码散列直接拒绝。

### 6.3 曲线与敏感性

```powershell
python -B -X utf8 code/Problem3/plot_results.py <OUTPUT_DIR> --png
python -B -X utf8 code/Problem3/sensitivity.py <OUTPUT_DIR> --data <DATA_DIR> --cases 44 46 90 --output results/Problem3/sensitivity/fixed_plan.csv
```

曲线要求每个核数覆盖相同用例集合。敏感性脚本只在内存中改变 L2 参数，
固定方案和正式配置不变，输出不应并入正式成绩。

## 7. 验证流程与回归原则

候选阶段的 `Evaluator` 只用于加速等价官方精评；最终验收始终执行：

1. 写出标准两字段 JSON，再重新读取并比较 `Plan`。
2. 用原版问题三评测器核对 Makespan、搬运、内存、Cache 统计和操作时间线。
3. 用原版问题二评测器评同一方案，取得无 L2 Makespan 和对照搬运量。
4. 确认两种配置的 COPY 搬运统计一致，并记录输入、配置、源码和方案 SHA-256。

批量 `summarize()` 还检查增加可用核心后保底成绩不退化。
`compare()` 对每个相同用例/核数输出旧新 Makespan、额外字节、无 L2 时间、Cache 比值，
并分别统计四项非退化条件。

候选精评失败、结构非法、预算耗尽或追加阶段没有通过守卫时，程序保留进入阶段的方案。
因此一次优化可以没有改善，但不能因追加阶段使固定基准恶化。

## 8. 结果目录约定

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 输入、配置、源码散列、软件版本和运行参数 |
| `summary.json` | 完成状态、失败用例、调用次数、曲线和比较统计 |
| `metrics.csv` | 逐用例、逐核的完整数值指标 |
| `curves.csv` | 相同用例集合的核数均值 |
| `comparison.csv` | 与基准逐配置比较和非退化布尔值 |
| `case_*/core_N/*_multicore_res.json` | 规定格式的提交方案 |
| `official.json` | 有 L2/无 L2 摘要、方案散列和最终核验标志 |
| `diagnostics.json` | Cache miss、带宽需求、Pipe 工作和跨核等待 |
| `search.json` | 基础搜索候选、下界、预算和精评记录 |
| `refinement.json` | 时间线追加候选、固定基准、守卫和追加耗时 |

实验数据应与生成它的源码、配置和输入图散列一起保存。不同配置、不同官方评测器版本、
不同用例集合的曲线不能直接拼接为同一平均结果。

## 9. 边界与阅读顺序

有限候选搜索不能证明一般输入图的全局最优；安全下界只用于排除不可能优于当前时间的候选。
Cache 命中率较高不保证 Makespan 下降，新增核心也不保证每个独立方案严格变快。
程序通过空核心补齐和固定基准守卫保留已知较优解，但不同机器的软预算截断点可能不同。

建议阅读顺序为：`Problem3/README.md` 的模型和验收章节、`solve.py` 的流程、
`model.py` 的代理与候选、`search.py` 的邻域、`tools/preview.py` 的精评复用，
最后对照题包官方评测实现检查事件语义。

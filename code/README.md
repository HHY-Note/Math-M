# 通用神经网络处理器多核调度：代码说明

本目录按“通用工具 + 三个问题”组织。本文说明提交目录内各文件的作用、
运行依赖和使用方法；各题的建模与算法说明见对应目录的 README。

## 1. 目录与实现状态

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

| 部分 | 当前提供的内容 | 详细说明 |
| --- | --- | --- |
| 通用工具 | 图处理、配置读取、JSON 读写和官方接口适配 | 本文第 2 节 |
| 问题一 | 完整求解入口、批量评测和绘图程序 | [问题一：模型与算法](Problem1/README.md) |
| 问题二 | 从原图独立求解场景 B、官方评分及批量实验 | [问题二：数学模型、算法与实验分析](Problem2/README.md) |
| 问题三 | 独立 Cache 求解、原版双配置核验、结构覆盖实验及绘图 | [问题三](Problem3/README.md) |

三个问题均可独立从原图运行，不读取其他问题的方案、成绩或中间文件。
通用工具只共享图处理、方案表示、官方接口及实验输出能力。
目录采用 Python 命名空间包，求解入口会将本目录加入模块搜索路径。

## 2. 每个文件的职责

### 2.1 通用工具

| 文件 | 主要接口 | 职责 |
| --- | --- | --- |
| `tools/graph.py` | `Graph`、`topological()` | 校验原图，建立算子 DAG 和张量关系，计算拓扑序、弱连通分量、安全链、深度和结构统计 |
| `tools/plan.py` | `Plan`、`check_plan()`、`arrange()`、`runs()` | 场景 B 通用方案表示、子图 DAG 和核内顺序检查，不含各题搜索算法 |
| `tools/official.py` | `Settings.read()`、`Settings.evaluate()`、`read_json()`、`write_json()` | 读取官方参数、严格读取 JSON、调用官方核内调度及第一问评测器、写出 UTF-8 JSON |
| `tools/official.py` | `SceneBSettings` | 读取场景 B 参数、调用第二问评测器及整图单核基准 |
| `tools/official.py` | `CacheSettings` | 官方 Cache 配置与第三问评测，同方案无 L2 对照 |
| `tools/preview.py` | `evaluate()` | 场景 B 两种配置的候选精评加速；原版官方入口另做最终完整核验 |
| `tools/experiment.py` | `digest()`、`official_summary()`、`write_csv()` | 共用实验输出 |
| `tools/plot_results.py` | `plot()` | 共用平均加速比绘图；`--problem 2` 选择第二问标题 |

`Graph` 保留两种互补的信息：算子 DAG 用于判断执行先后，张量及其生产者、
消费者集合用于计算共享输入和切图搬运量。节点 ID 可以稀疏、不连续。
图的结构分类只用于覆盖统计，不使用用例编号指导求解。

`Settings` 集中保存容量、DDR 带宽和场景 A 的等待周期。`official.py`
是接口适配层，不包含一份重新编写的官方评测算法。其中图处理和 JSON 接口
可以复用，`Settings.evaluate()` 当前接入的仍是问题一评测器。

### 2.2 问题一程序

| 文件 | 主要接口 | 职责 |
| --- | --- | --- |
| `Problem1/solve.py` | `Solver`、`Solution`、`main()` | 组织候选搜索，保留最好方案，逐核数求解，执行最终官方核验并输出方案 |
| `Problem1/partition.py` | `Partition`、`describe()`、`candidates()`、`refinements()`、`feedback()`、`bottleneck_bands()` | 描述划分、计算代理成本，生成结构候选、拆分合并候选和瓶颈 Task 内的分层拆分 |
| `Problem1/scheduling.py` | `list_schedule()`、`lower_bound()`、`neighbors()`、`schedule_cost()`、`ranked_neighbors()` | 列表调度、时间下界、合法迁移及最终调度候选排序 |
| `Problem1/execution.py` | `Kernel`、`Compiled`、`Engine`、`check_schedules()`、`assert_official()` | 按张量索引构造局部图，复用子图核内展开，执行精确多核模拟及逐操作核验 |
| `Problem1/benchmark.py` | `run_case()`、`verify()`、`main()` | 批量求解，汇总逐用例指标和平均加速比，记录复现信息，支持多进程与对照核验 |
| `Problem1/plot_results.py` | `plot()` | 从批量指标生成 1～5 核平均加速比曲线及曲线数据，可导出 SVG、PNG |

### 2.3 文档

| 文件 | 内容 |
| --- | --- |
| `README.md` | 总体目录、逐文件职责、依赖、命令和方案格式 |
| `Problem1/README.md` | 按建模论文结构说明第一问的假设、符号、模型、算法、实现和验证方法 |
| `Problem2/README.md` | 第二问代码职责、运行方法及实验指标说明 |
| `Problem3/README.md` | 第三问模型、独立算法、核验方式、实验与运行命令 |

## 3. 运行依赖

### 3.1 Python 与绘图库

- Python 3.10 或以上。
- 求解器和批量评测只使用 Python 标准库。
- 绘制 SVG 使用 ReportLab；选择 `--png` 时还使用 pypdfium2。

需要绘制 SVG 时，安装验证过的 ReportLab 版本：

```text
python -m pip install reportlab==4.4.9
```

需要额外导出 PNG 时，再安装：

```text
python -m pip install pypdfium2==5.13.0
```

这些库仅由绘图脚本使用。

### 3.2 官方题包

**官方评测源码、测试数据和 `config.txt` 由赛题材料提供，不包含在本提交目录中。**

`tools/official.py` 默认从提交目录同级的 `题目/code` 加载官方模块。
请按这一位置提供原始官方代码；如评测环境采用其他位置，调整该文件顶部的
`REFERENCE` 路径即可，保留官方程序和参数不变。

问题一实际调用的官方模块为：

| 官方模块 | 调用用途 |
| --- | --- |
| `contest_io.py` | 拒绝重复键的 JSON 读取 |
| `evaluation_validation.py` | 配置读取、原图及执行合法性校验 |
| `multicore_cut_evaluate_problem_1.py` | Task 构建、边界 COPY 和第一问精确评测 |
| `schedule_step1.py` | 核内操作访问顺序 |
| `schedule_step2.py` | 容量管理、换出和换入 |
| `schedule_step3.py` | 多 Pipe 调度、内存复用依赖和执行图 |
| `stub_multicore_cut_and_schedule.py` | 方案校验和依赖图辅助函数，不使用其中的随机方案生成器作为算法 |

单用例默认从输入图同目录读取 `config.txt`，可用 `--config` 明确指定。
批量评测从 `--data` 指定的数据目录读取原始 `config.txt`。
程序不会用写死在算法中的参数替代缺失的官方配置。

## 4. 运行方法

以下命令在本 `code` 目录执行。大写名称是占位符，使用时替换为实际路径；
结果路径宜指定在提交代码目录之外。

### 4.1 求解单个用例

```text
python -B -X utf8 Problem1/solve.py GRAPH.json -n 4 --config CONFIG.txt -o CASE_multicore_res.json
```

`CASE` 应与输入图的文件名去掉 `.json` 后一致。程序在写出方案前调用官方
评测器，核对完工时间、搬运量、内存峰值以及全部 Task、操作的开始和结束时间。
同名的 `*_search.json` 单独记录搜索和核验信息。

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `graph` | 输入计算图 JSON | 必填 |
| `-n` / `--num-cores` | 核数，范围 1～5 | 必填 |
| `--config` | 官方固定配置 | 输入图同目录的 `config.txt` |
| `-o` / `--output` | 方案文件路径 | 程序自动生成，位于提交代码目录之外 |
| `--budget` | 每种核数的搜索软时间预算，单位秒 | 180 |
| `--max-candidates` | 粗粒度阶段和结构细化阶段各自的精评候选上限 | 3 |
| `--moves` | 任务迁移候选上限 | 8 |

单核采用整图基线。求解 4 核时先求解 2、3 核，以复用已有可行方案。
预算在候选之间检查，正在执行的核内展开及最终官方核验不被中断，
因此 `--budget` 不表示命令的总运行时间上限。

### 4.2 批量求解

```text
python -B -X utf8 Problem1/benchmark.py --data DATA_DIR --output OUTPUT_DIR --workers 2
```

**不指定 `--cases` 时，运行数据目录内全部 `case_编号.json`。**
完整官方数据目录应包含 100 个用例。默认计算单核基线和 2、3、4、5 核方案。
小范围运行可以明确指定用例编号：

```text
python -B -X utf8 Problem1/benchmark.py --data DATA_DIR --output OUTPUT_DIR --cases 1 14 --workers 2
```

批量入口还支持 `--cores`、`--budget`、`--max-candidates`、`--moves`。
`--cores` 必须是 2～5 中递增且不重复的核数；正式生成完整曲线时保留默认值。
`--compare COMPARE_DIR` 用于与同条件的完整评测结果对照；对相同方案复用已核验
结果前，会核对输入、配置、官方源码和 Python 版本，并再次核对模拟结果。

批量输出的 `metrics.csv` 含逐用例 Makespan、加速比、额外搬运量、换出换入量
和运行耗时；`summary.json` 汇总平均指标；`manifest.json` 记录运行参数及哈希。
输出方案、评测摘要和搜索记录按用例与核数组织。

### 4.3 生成加速比曲线

```text
python -B -X utf8 Problem1/plot_results.py OUTPUT_DIR
python -B -X utf8 Problem1/plot_results.py OUTPUT_DIR --png
```

程序读取 `metrics.csv`，生成 `mean_speedup.svg`、`mean_speedup.csv`；
`--png` 额外生成 `mean_speedup.png`。各核数必须覆盖同一用例集合。
平均加速比按“先计算每个用例的加速比，再取算术平均”统计，单核固定为 1。

## 5. 输入输出约定

输入图使用官方 `ops`、`tensors`、`edges` 三个列表。所有节点 ID 唯一，
`cycles` 和 `size` 非负，原始依赖图必须无环。

方案顶层必须且只能含两个字段。以下是仅有操作 `11` 时的两核格式示例：

```json
{
  "node_to_subgraph": {"11": 0},
  "core_schedules": [[0], []]
}
```

`node_to_subgraph` 覆盖全部非 `COPY_IN` / `COPY_OUT` 操作，且每个操作只出现一次。
`core_schedules` 外层长度等于指定核数，每个子图恰好安排一次，允许空核心。
非法覆盖、依赖环、缺失配置等情况会报错；精确模拟与官方核验不一致时不会
写出通过验收的新方案。详细算法与约束说明见各题 README。

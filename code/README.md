# 比赛代码与全量运行

本目录包含三个问题的独立求解程序。每题从原始计算图和题目配置开始求解，输出比赛规定的多核划分与调度方案；三题之间不读取彼此的方案或搜索结果。各题的建模、算法和实现细节分别见其目录下的 README。

## 代码结构

| 目录或文件 | 职责 |
| --- | --- |
| `Problem1/` | 场景 A 的划分、调度、精确模拟与全量运行 |
| `Problem2/` | 场景 B 的划分、调度与全量运行 |
| `Problem3/` | 共享只读 L2 Cache 场景的建模、搜索、诊断与全量运行 |
| `tools/graph.py` | 原图校验、算子 DAG 和张量关系 |
| `tools/plan.py` | 场景 B/C 的方案表示与合法性检查 |
| `tools/official.py` | 读取题目配置，调用题包官方评测器 |
| `tools/preview.py` | 问题二、三搜索阶段的等价精评复用 |
| `tools/experiment.py` | 文件散列、官方结果摘要和 CSV 写出 |

题包官方程序位于 `题目/code/`，由代码只读调用。提交方案的 JSON 顶层仅有 `node_to_subgraph` 和 `core_schedules` 两个字段；每份最终方案均调用对应官方评测器核验。

## 输入与运行环境

- 在项目根目录 `数模/` 执行以下命令，使用 Python 3.10 或更高版本。求解与全量运行仅依赖 Python 标准库和题包官方程序。
- `--data` 指向含 `config.txt` 及 `case_001.json` 至 `case_100.json` 的目录。三题均固定运行这 100 个用例。
- `--output` 指向不存在或为空的目录。每题使用独立目录，防止混入其他运行结果。
- `--workers` 表示同时处理的用例数，与题目方案中的核心数无关。下面统一使用 5 个工作进程；可按机器内存调整。

## 唯一全量运行入口

**问题一：100 个用例，各求解 2～5 核，并计算单核基线。**

```powershell
python -X utf8 -B code/Problem1/benchmark.py --data 题目/data --output results/Problem1_full --workers 5
```

**问题二：100 个用例，各求解 2～5 核，并计算单核基线。**

```powershell
python -X utf8 -B code/Problem2/benchmark.py --data 题目/data --output results/Problem2_full --workers 5
```

**问题三：100 个用例，各求解 1～5 核；每份方案均在有、无只读 L2 Cache 两种配置下核验。**

```powershell
python -X utf8 -B code/Problem3/benchmark.py --data 题目/data --output results/Problem3_submission_full --workers 5
```

问题一、二预期各生成 **400** 条多核记录；问题三预期生成 **500** 条记录。运行结果中的 `summary.json` 应显示 `completed: true`、对应的 `rows` 数量，并且 `errors` 或 `failures` 为空。进程退出码为 0 才表示全量运行完成。

## 输出与核验

每个输出目录的 `manifest.json` 记录输入图、配置、求解源码及官方源码的散列；`metrics.csv` 保存逐用例、逐核指标，`summary.json` 保存完成状态和汇总统计。每个 `case_XXX/core_N/` 下的 `case_XXX_multicore_res.json` 是对应核数的标准提交方案，`official.json` 是官方核验结果。问题三另外保存同一方案的有 L2 与无 L2 评分及 Cache 诊断，便于区分硬件 Cache 收益与划分调度收益。

问题一、二的单核整图结果用作加速比基线；问题三以无 L2 的整图单核时间作为统一基准。

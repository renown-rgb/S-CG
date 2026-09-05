# S-CG 数值实验复现指南（Reproducibility Guide）

本文档说明如何复现论文《…》中的全部数值实验。所有命令均在**项目根目录**（本目录）下执行。

## 0. 环境

| 组件 | 版本 |
|---|---|
| Python | 3.9.13 |
| NumPy | 2.0.2 |
| SciPy | 1.13.1 |
| Matplotlib | 3.9.4 |
| gurobipy（可选） | 12.0.3 |
| tushare / pandas（可选） | 仅投资组合实验 |

```bash
pip install -r requirements.txt
```

**Gurobi 说明**：S-CG 核心求解器只用 NumPy + SciPy，**不依赖 Gurobi**。
只有「对比 Gurobi 基线」这一步需要商业 Gurobi license；没有 license 时用 `--no-gurobi` 即可跳过，
并仍可复现 S-CG 与 HiGHS 的全部对比。

> ✅ 论文正文与 `.bib` 中 Gurobi 版本均已统一为 **12.0**（与本机实测安装 **12.0.3** 一致）。

## 1. 冒烟测试（30 秒内确认环境可用）

```bash
python -m pytest tests/test_fast_solver.py -v        # 16 个单元测试，应全部通过
python experiments/experiment_runner.py --quick --no-gurobi
```

## 2. 实验 → 命令 → 输出 → 论文对应

| 实验 | 命令 | 输出 CSV | 论文表/图 |
|---|---|---|---|
| **可扩展性**（主实验，m=10, τ=0.1, 10 trials） | `python experiments/experiment_runner.py --n 100 1000 10000 100000 1000000 --m 10 --tightness 0.1 --trials 10 --solvers S-CG Gurobi HiGHS --output experiments/results/scalability_final.csv` | `scalability_final.csv` | `tab:scalability`、图 2/3/4 |
| **消融**（组件贡献，m=10, τ=0.05） | `python experiments/run_ablation.py` | `ablation.csv` | `tab:ablation`、图 5 |
| **退化相变 + 密度敏感性** | `python experiments/run_sensitivity.py` | `degeneracy.csv`, `density.csv` | `tab:phase_bounds`、图 6 |
| **投资组合**（CSI 300，两年日线 2023-06→2025-05） | `python experiments/run_portfolio.py` | `portfolio_real.csv`（数据 `csi300_daily_2y.csv`） | `tab:portfolio` |
| **投资组合**（小 m 大 n，A 股 2000 只，约 60 交易日） | `python experiments/run_portfolio_small_m.py` | `portfolio_small_m.csv`（数据 `a_share_daily.csv`） | `tab:portfolio_small_m` |
| **统计显著性检验**（Wilcoxon 配对 + Cohen's d） | `python experiments/run_statistics.py` | `statistics.csv` | `tab:statistics` |
| **cvxopt 对比**（稠密 IPM，n≤10^4；n=5×10^4 稀疏 OOM） | `python experiments/run_cvxopt_compare.py` | `cvxopt_compare.csv` | cvxopt 讨论段落 |
| **开源求解器对比**（HiGHS dual simplex vs HiGHS-IPM） | `python experiments/run_opensource.py` | `opensource_compare.csv` | `tab:opensource` |
| **内存测量**（S-CG CSR/CSC 精确字节数） | `python experiments/run_memory.py` | `memory.csv` | `tab:memory` |
| **ADMM 对比** | `python experiments/run_admm_compare.py` | `admm_comparison.csv` | ADMM 对比相关表 |
| **显式稀疏约束** | `python experiments/compare_sparse_solvers.py` | `sparse_compare.csv` | `tab:explicit_check` |

无 Gurobi 时，主实验命令追加 `--no-gurobi`（自动移除 Gurobi，只跑 S-CG + HiGHS）。

> 注意：`experiment_runner.py` 默认输出到 `results/experiment_a.csv`，但绘图脚本读取的是
> `scalability_final.csv`，因此复现论文图 2/3/4 时务必用 `--output experiments/results/scalability_final.csv`。

## 3. 生成论文图表

```bash
python experiments/plot_figures.py    # fig2–fig7 + 复杂度边界图 → paper/figures/
python experiments/perf_profile.py    # Dolan-Moré 性能剖面 → paper/figures/fig_perf_profile_all.pdf
```

`perf_profile.py` 读取 `experiments/results/scalability_with_gurobi.csv`（636 个实例的性能剖面，对应论文 `fig:perf`）。

## 4. 复现性保证

- **随机种子固定**：所有问题生成用 `np.random.RandomState(seed)`，seed 由确定性公式给出（如
  `seed = 1000 * log10(n) + trial`），因此每次运行生成**完全相同**的问题实例。
- **Gurobi 参数固定**：`OutputFlag=0, Method=-1（自动）, Threads=1, TimeLimit=300`（见 `experiment_runner.py` 的 `solve_gurobi`）。
- **计时取中位数**：含求解器计时的实验（可扩展性、开源对比、投资组合等）对同一实例重复多次求解并取中位数计时（投资组合实验 `N_TRIALS=7`），避免单次运行噪声。
- **结果已存盘**：`experiments/results/` 下所有 CSV 均随代码提供，不重跑也能据此复算论文所有表格。
- **遗留 method 字段**：可扩展性 CSV（`scalability_final.csv`、`scalability_with_gurobi.csv` 等）中 `n=100` 行的 `method` 列为旧版代码的 `full_lp`（旧版对 `n<1000` 直接短路至完整 LP）。当前两级设计中 `n=100` 改由 Stage 2 列生成求解，目标值、耗时与 Stage-1 命中率（0%）均不变，论文表格数值不受影响；该字段仅属历史命名。
- **确定性脚本**：`run_statistics.py`、`run_memory.py` 是纯重算（无随机性、无求解器），从已存盘的 CSV
  直接复算 `tab:statistics`、`tab:memory`，结果**逐字节确定**。
- **无 Gurobi 也可复现的部分**：`run_statistics.py`、`run_cvxopt_compare.py`、`run_opensource.py`、
  `run_memory.py` 运行本身均不调用 Gurobi；其中 `run_statistics.py` 的输入
  `scalability_with_gurobi.csv` 需 Gurobi 生成一次，但**从该 CSV 复算 p 值/效应量无需 Gurobi**。

## 5. 目录结构（提交包包含）

```
fast_solver.py          # 核心求解器（两级：KKT 证书 → 列生成精确求解）
final_benchmark.py      # GA 基线 + 问题生成器
solver/                 # 显式稀疏约束求解器
experiments/            # 实验框架 + results/*.csv
tests/                  # 16 个单元测试
docs/                   # 算法/证明文档
README.md               # 项目说明
requirements.txt        # 依赖锁定
REPRODUCE.md            # 本文档
```

未包含：`legacy/`（旧版 GA 探索代码）、`data_analysis/`（无关的聚类分析）、`paper/`（论文源文件，另随投稿提交）。

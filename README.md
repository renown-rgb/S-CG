# S-CG: 稀疏单纯形线性规划的快速求解

## 核心结果

S-CG在 $m \ll n$ 的稀疏单纯形 LP 上比 Gurobi 快 **208 倍**（$n=10^6$, $m=10$），
全部结果达到机器精度。

## 项目结构

```
d:/dxxishu/
├── fast_solver.py          # ★ 核心求解器 (两级架构: KKT证书→列生成精确求解)
├── final_benchmark.py      # GA基线 + 问题生成器
│
├── experiments/            # 实验框架 + 结果
│   ├── experiment_runner.py    # 统一实验入口
│   ├── problem_factory.py      # 受控问题生成
│   ├── stats_utils.py          # 统计 + LaTeX表格生成
│   ├── plot_figures.py         # 论文图表生成
│   ├── perf_profile.py         # Dolan-Moré性能剖面
│   ├── run_ablation.py         # 消融实验
│   ├── run_portfolio.py        # 投资组合优化 (tushare真实A股)
│   └── results/                # ★ 所有实验CSV结果
│
├── tests/                  # 单元测试
│   └── test_fast_solver.py # 16个测试用例
│
├── paper/                  # 论文源文件（LaTeX 另随投稿提交）
│   └── figures/            # 图表 (PDF+PNG)
│
├── docs/                   # 文档
│   ├── algorithm_framework.md  # 算法逻辑框架 (Markdown)
│   ├── degeneracy_proof.md     # 退化证明详解
│   └── two_sparse_proof.md     # 双列退化证明
│
└── legacy/                 # 旧版代码(参考)
    ├── iga_simplex.py      # Phase 1: 连续GA (失败)
    ├── zero_one_ga.py      # Phase 2: 0-1 GA
    ├── hard_problem.py     # 困难问题生成
    └── ...
```

## 快速开始

```bash
# 运行单元测试
python -m unittest tests.test_fast_solver -v

# S-CG vs Gurobi 快速对比
python -c "
from fast_solver import FastSolver
from final_benchmark import generate_problem
prob = generate_problem(n=100000, m=10, seed=500)
r = FastSolver(False).solve(prob['c'], prob['A_csr'], prob['b'])
print(f'S-CG: {r[\"objective\"]:.6f} [{r[\"method\"]}] {r[\"time\"]*1000:.1f}ms')
"
```

## 运行实验

```bash
# 可扩展性实验
python experiments/experiment_runner.py --n 100000 1000000 --m 10 --trials 10 --solvers S-CG Gurobi HiGHS

# 消融实验
python experiments/run_ablation.py

# 投资组合优化 (需要tushare token)
python experiments/run_portfolio.py

# 生成论文图表
python experiments/plot_figures.py
```

## 论文

论文 LaTeX 源文件随投稿另行提交，不包含在本代码仓库中。全部数值实验的复现步骤见 [REPRODUCE.md](REPRODUCE.md)。

## 依赖

精确依赖版本见 [requirements.txt](requirements.txt)。核心求解器仅需 NumPy + SciPy：
- Gurobi + gurobipy（可选，用于对比实验；无 Gurobi 时用 `--no-gurobi`）
- tushare（可选，仅当重新联网抓取行情数据时需要；数据已随包提供为 CSV）

## 引用

论文正在投稿中，发表后请引用（`journal`/`year` 发表时补全）：

```bibtex
@article{he_scg,
  author = {He, Mingpeng and Ding, XinNan and Chen, Fangfang and Hu, Qiying and Jiang, Zhongyi},
  title  = {Optimal properties and Algorithm for large-scale linear programming problem with a probability simplex constraint and inequality linear constraints},
  note   = {under review}
}
```

## 许可证

本项目采用 [MIT License](LICENSE)。

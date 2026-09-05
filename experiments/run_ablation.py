"""
================================================================================
消融实验 (Experiment C) — 量化各组件贡献

S-CG 消融变体:
  1. S-CG 无 Level 1 (始终进入 Level 2 列生成)
  2. S-CG 无稀疏定价 (用 dense 定价)
  3. HiGHS 直接求解 (独立基线, 非 S-CG 组件)
  4. Full S-CG (完整版, 基线)
================================================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import linprog
from scipy import sparse
import time
import csv
from typing import Dict

from fast_solver import FastSolver, ColumnGenerationSolver, SingleColumnSolver
from experiments.problem_factory import generate_problem
from experiments.stats_utils import group_by, summarize


# ============================================================================
# 消融变体
# ============================================================================

class FastSolver_NoLevel1:
    """消融: 跳过 Level 1, 直接进入 Level 2"""
    def __init__(self):
        self.cg_solver = ColumnGenerationSolver(verbose=False)

    def solve(self, c, A, b):
        n = len(c)
        t0 = time.perf_counter()
        # 跳过 Level 1
        cg_result = self.cg_solver.solve(c, A, b)
        cg_result['time'] = time.perf_counter() - t0
        cg_result['method'] = 'column_generation_noL1'
        return cg_result


class FastSolver_DensePricing:
    """消融: 用 dense 定价 (O(mn) 而非 O(nnz))"""
    def __init__(self):
        self.cg_solver = ColumnGenerationSolver(verbose=False)

    def solve(self, c, A, b):
        n = len(c)
        t0 = time.perf_counter()
        # 强制 dense: 转换 A 为 dense
        if sparse.issparse(A):
            A_dense = A.toarray()
        else:
            A_dense = A
        cg_result = self.cg_solver.solve(c, A_dense, b)
        cg_result['time'] = time.perf_counter() - t0
        cg_result['method'] = 'column_generation_dense'
        return cg_result


class HiGHS_Direct:
    """基线: HiGHS 直接求解完整 LP (非 S-CG 组件, 用于对比)"""
    def solve(self, c, A, b):
        n = len(c)
        t0 = time.perf_counter()
        if n >= 100000 and sparse.issparse(A):
            A_eq = sparse.csr_matrix(([1.0]*n, ([0]*n, range(n))), shape=(1, n))
            res = linprog(c, A_ub=A, b_ub=b,
                          A_eq=A_eq, b_eq=np.array([1.0]),
                          bounds=(0, None), method='highs')
        else:
            A_u = A.toarray() if sparse.issparse(A) else A
            res = linprog(c, A_ub=A_u, b_ub=b,
                          A_eq=np.ones((1, n)), b_eq=np.array([1.0]),
                          bounds=(0, None), method='highs')

        elapsed = time.perf_counter() - t0
        if res.success:
            x = res.x.copy(); x[x < 1e-14] = 0; x /= x.sum()
        else:
            x = np.zeros(n)
        return {'x': x, 'objective': float(c @ x),
                'sparsity': int(np.sum(x > 1e-14)),
                'time': elapsed, 'method': 'highs_direct'}


# ============================================================================
# 消融实验
# ============================================================================

VARIANTS = {
    'S-CG_full': FastSolver(verbose=False),
    'S-CG_noL1': FastSolver_NoLevel1(),
    'S-CG_dense': FastSolver_DensePricing(),
    'HiGHS_direct': HiGHS_Direct(),
}


def run_ablation(n_list=[1000, 10000, 100000, 500000],
                 m=10, tightness=0.05, n_trials=10):
    """消融实验主流程"""
    results = []

    for n in n_list:
        for trial in range(n_trials):
            seed = 1000 * int(np.log10(max(n, 10))) + trial
            prob = generate_problem(n=n, m=m, seed=seed,
                                    slack_tightness=tightness)
            c, A_csr, b = prob['c'], prob['A_csr'], prob['b']

            for vname, solver in VARIANTS.items():
                t0 = time.perf_counter()
                try:
                    result = solver.solve(c, A_csr.copy(), b)
                    results.append({
                        'n': n, 'm': m, 'tightness': tightness,
                        'trial': trial, 'variant': vname,
                        'objective': result['objective'],
                        'time': result['time'],
                        'sparsity': result['sparsity'],
                        'method': result.get('method', ''),
                    })
                except Exception as e:
                    print(f"  ERROR [{vname}] n={n} trial={trial}: {e}")
                    results.append({
                        'n': n, 'm': m, 'tightness': tightness,
                        'trial': trial, 'variant': vname,
                        'objective': np.inf, 'time': np.inf,
                        'sparsity': -1,
                    })

        print(f"  n={n}: {n_trials} trials complete "
              f"({len(VARIANTS)} variants each)")

    return results


def ablation_summary(results):
    """消融摘要"""
    grouped = group_by(results, 'n', 'variant')
    print("\n" + "=" * 70)
    print("Ablation Study Summary")
    print("=" * 70)
    print(f"{'n':>8} | {'Variant':>14} | {'Mean Time':>10} | {'Speedup':>8}")
    print("-" * 50)

    baseline_times = {}
    for key, group in grouped.items():
        n, variant = key
        times = [r['time'] for r in group if not np.isinf(r['time'])]
        if not times:
            continue
        mean_t = np.mean(times)
        if variant == 'S-CG_full':
            baseline_times[n] = mean_t

        speedup = baseline_times.get(n, mean_t) / mean_t if mean_t > 0 else 0
        print(f"{n:>8,} | {variant:>14} | {mean_t:8.4f}s | {speedup:7.1f}x")

    return grouped


if __name__ == "__main__":
    print("Running ablation study...")
    results = run_ablation(
        n_list=[1000, 10000, 100000],
        m=10, tightness=0.05, n_trials=5)

    ablation_summary(results)

    # Save
    os.makedirs('experiments/results', exist_ok=True)
    keys = results[0].keys()
    with open('experiments/results/ablation.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f"\nResults saved to experiments/results/ablation.csv")

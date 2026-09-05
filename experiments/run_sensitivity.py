"""
================================================================================
敏感性分析 (Experiments E, F)

E: 退化相变 — slack_tightness 从 1e-4 到 1.0
   观察 Level 1 成功概率的相变
F: 密度敏感性 — A 密度从 0.1% 到 50%
   验证定价时间 O(nnz)
================================================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import linprog
from scipy import sparse
import time
import csv
from collections import defaultdict

from fast_solver import FastSolver
from experiments.problem_factory import generate_problem, diagnose_problem


# ============================================================================
# Experiment E: 退化相变
# ============================================================================

def run_degeneracy_analysis(n_list=[1000, 10000, 100000, 500000],
                             m=10, n_trials=20):
    """
    扫描 slack_tightness, 统计 Level 1 成功概率
    """
    tightness_values = np.logspace(-4, 0, 15)  # 1e-4 to 1.0
    results = []

    for n in n_list:
        print(f"\n--- n={n:,} ---")
        for tightness in tightness_values:
            l1_success = 0
            total_time = 0

            for trial in range(n_trials):
                seed = 10000 + trial
                prob = generate_problem(n=n, m=m, seed=seed,
                                        slack_tightness=float(tightness))
                c, A, b = prob['c'], prob['A_csr'], prob['b']

                solver = FastSolver(verbose=False)
                result = solver.solve(c, A, b)

                if result['method'] == 'single_column_verified':
                    l1_success += 1
                total_time += result['time']

            success_rate = l1_success / n_trials
            results.append({
                'n': n, 'tightness': float(tightness),
                'l1_success_rate': success_rate,
                'mean_time': total_time / n_trials,
            })
            print(f"  tightness={tightness:.4f}: L1={success_rate:.0%} "
                  f"time={total_time/n_trials:.4f}s")

    return results


# ============================================================================
# Experiment F: 密度敏感性
# ============================================================================

def run_density_analysis(n=100000, m=10, n_trials=10):
    """
    扫描 A 密度, 验证定价时间 O(nnz)
    """
    density_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    results = []

    print(f"\n--- Density Analysis (n={n:,}) ---")
    for density in density_values:
        pricing_times = []
        nnz_values = []

        for trial in range(n_trials):
            seed = 20000 + trial
            prob = generate_problem(n=n, m=m, seed=seed, density=density,
                                    slack_tightness=0.1)
            c, A, b = prob['c'], prob['A_csr'], prob['b']

            nnz = A.nnz
            nnz_values.append(nnz)

            solver = FastSolver(verbose=False)
            result = solver.solve(c, A, b)
            pricing_times.append(result['time'])

        results.append({
            'n': n, 'density': density,
            'nnz_mean': np.mean(nnz_values),
            'mean_time': np.mean(pricing_times),
            'time_per_nnz': np.mean(pricing_times) / max(np.mean(nnz_values), 1),
        })
        print(f"  density={density:.3f} nnz={np.mean(nnz_values):,.0f} "
              f"time={np.mean(pricing_times):.4f}s "
              f"t/nnz={np.mean(pricing_times)/max(np.mean(nnz_values),1):.2e}")

    return results


# ============================================================================
# 主流程
# ============================================================================

if __name__ == "__main__":
    os.makedirs('experiments/results', exist_ok=True)

    print("=" * 60)
    print("Experiment E: Degeneracy Phase Transition")
    print("=" * 60)
    e_results = run_degeneracy_analysis(n_trials=10)

    with open('experiments/results/degeneracy.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=e_results[0].keys())
        w.writeheader()
        w.writerows(e_results)

    print("\n" + "=" * 60)
    print("Experiment F: Density Sensitivity")
    print("=" * 60)
    f_results = run_density_analysis(n_trials=5)

    with open('experiments/results/density.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=f_results[0].keys())
        w.writeheader()
        w.writerows(f_results)

    print("\nAll sensitivity results saved to experiments/results/")

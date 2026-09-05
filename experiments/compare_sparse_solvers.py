"""
================================================================================
两个稀疏求解器全面对比

Solvers:
  1. ExplicitSparse:  min cᵀx, Ax≤b, Σx=1, x≥0, ‖x‖₀≤K
  2. GeneralSparse:   Ax≤b, ‖x‖₀≤K, x∈ℝⁿ (无限制)

Baseline: Gurobi MIP
================================================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'solver'))

import numpy as np
from scipy.optimize import linprog
from scipy import sparse
import time, csv
from collections import defaultdict

from fast_solver import FastSolver
from solver.explicit_sparse import ExplicitSparseSolver
from solver.general_sparse import GeneralSparseSolver


def generate_general_problem(n, m, seed=42, density=0.05, b_scale=0.5):
    """生成通用测试实例 (无单纯形约束, x 无符号限制)"""
    rng = np.random.RandomState(seed)

    # 稀疏约束矩阵
    A = sparse.random(m, n, density=density, format='csr',
                      data_rvs=lambda sz: rng.randn(sz),
                      random_state=rng)

    # 真解: K_sparse 个随机非零元
    K_true = min(m + 1, n // 20)
    x_true = np.zeros(n)
    nz = rng.choice(n, K_true, replace=False)
    x_true[nz] = rng.randn(K_true) * 2

    # b = A @ x_true + slack (确保可行)
    b = A @ x_true + np.abs(rng.randn(m)) * b_scale + 0.1
    c = rng.randn(n) * 2

    return {'A_csr': A, 'A_csc': A.tocsc(), 'b': b, 'c': c, 'n': n, 'm': m,
            'x_true': x_true, 'true_nnz': K_true}


def benchmark():
    configs = [
        # (n, m, K, density, b_scale, n_trials)
        (100,  10, 3, 0.10, 0.5, 10),
        (100,  20, 5, 0.15, 0.5, 10),
        (200,  15, 4, 0.08, 0.5, 10),
        (200,  30, 6, 0.12, 0.5, 10),
        (500,  20, 5, 0.03, 0.5, 10),
        (500,  40, 8, 0.05, 0.5, 10),
        (1000, 20, 5, 0.02, 0.5, 10),
        (1000, 50, 8, 0.03, 0.5, 10),
    ]

    results = []
    print("=" * 85)
    print("  ExplicitSparse vs GeneralSparse vs Gurobi MIP")
    print("=" * 85)

    for n, m, K, density, b_scale, n_trials in configs:
        print(f"\n-- n={n}, m={m}, K={K}, density={density} --")

        for trial in range(n_trials):
            seed = 30000 + n + m * 100 + trial
            prob = generate_general_problem(n, m, seed, density, b_scale)
            A = prob['A_csr']; b = prob['b']; c = prob['c']

            row = {'n': n, 'm': m, 'K': K, 'density': density,
                   'trial': trial, 'true_nnz': prob['true_nnz']}

            # --- ExplicitSparse (simplex version) ---
            solver_exp = ExplicitSparseSolver(K=K, verbose=False)
            t0 = time.time()
            r_exp = solver_exp.solve(c, A, b)
            row['exp_time_ms'] = (time.time() - t0) * 1000
            row['exp_feasible'] = not np.isinf(r_exp.get('objective', np.inf))
            row['exp_nnz'] = r_exp.get('sparsity', -1)
            row['exp_method'] = r_exp.get('method', 'fail')

            # --- GeneralSparse (sign-free version) ---
            solver_gen = GeneralSparseSolver(K=K, objective='minimize',
                                              max_candidates=150, verbose=False)
            t0 = time.time()
            r_gen = solver_gen.solve(A, b, c)
            row['gen_time_ms'] = (time.time() - t0) * 1000
            row['gen_feasible'] = r_gen.get('feasible', False)
            row['gen_nnz'] = r_gen.get('sparsity', -1)
            row['gen_method'] = r_gen.get('method', 'fail')
            row['gen_obj'] = r_gen.get('objective', np.inf) if r_gen['feasible'] else np.inf

            # --- Gurobi MIP ---
            try:
                import gurobipy as gp
                model = gp.Model("cmp")
                model.setParam('OutputFlag', 0)
                model.setParam('Threads', 1)
                model.setParam('TimeLimit', 60)
                model.setParam('MIPGap', 1e-4)

                xv = model.addMVar(n, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
                xp = model.addMVar(n, lb=0.0, vtype=gp.GRB.CONTINUOUS)
                xm = model.addMVar(n, lb=0.0, vtype=gp.GRB.CONTINUOUS)
                model.addConstr(xv == xp - xm)
                z = model.addMVar(n, vtype=gp.GRB.BINARY)
                model.addConstr(xp + xm <= z)
                model.addConstr(z.sum() <= K)

                model.addMConstr(A, xv, '<', b)
                model.setObjective(c @ xv, gp.GRB.MINIMIZE)
                model.optimize()

                row['mip_time_s'] = time.time() - t0 if 't0' in dir() else 0
                row['mip_feasible'] = model.status == gp.GRB.OPTIMAL
                row['mip_obj'] = model.ObjVal if model.status == gp.GRB.OPTIMAL else np.inf
            except:
                row['mip_time_s'] = np.inf
                row['mip_feasible'] = None
                row['mip_obj'] = np.inf

            results.append(row)

            print(f"  [{trial:2d}] Explicit: {row['exp_feasible']} "
                  f"({row['exp_time_ms']:.1f}ms) [{row['exp_method']}] | "
                  f"General: {row['gen_feasible']} ({row['gen_time_ms']:.1f}ms) "
                  f"[{row['gen_method']}] | "
                  f"MIP: {row['mip_feasible']}")

    # ============== 汇总 ==============
    print("\n" + "=" * 85)
    print("  Summary: Explicit vs General")
    print("=" * 85)

    by_config = defaultdict(list)
    for r in results:
        by_config[(r['n'], r['m'], r['K'])].append(r)

    print(f"{'n,m,K':>12} | {'Explicit':>12} | {'General':>12} | {'Winner':>10}")
    print("-" * 55)

    for key in sorted(by_config.keys()):
        group = by_config[key]
        n, m, K = key

        exp_success = sum(1 for r in group if r['exp_feasible']) / len(group)
        gen_success = sum(1 for r in group if r['gen_feasible']) / len(group)
        exp_time = np.mean([r['exp_time_ms'] for r in group])
        gen_time = np.mean([r['gen_time_ms'] for r in group])

        if exp_success > gen_success + 0.1:
            winner = 'Explicit'
        elif gen_success > exp_success + 0.1:
            winner = 'General'
        elif exp_time < gen_time:
            winner = 'Explicit'
        else:
            winner = 'General'

        print(f"{n:>4},{m:>3},{K:>2} | {exp_success:.0%} {exp_time:5.1f}ms | "
              f"{gen_success:.0%} {gen_time:5.1f}ms | {winner}")

    # Save
    os.makedirs('experiments/results', exist_ok=True)
    with open('experiments/results/sparse_compare.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved to experiments/results/sparse_compare.csv")

    return results


if __name__ == "__main__":
    benchmark()

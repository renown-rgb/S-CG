"""
================================================================================
ADMM 对比实验 — 陈圣杰 (2018) vs S-CG

陈圣杰 ADMM: L1正则化 + 交替方向乘子法
  min cᵀx + λ·||x||₁
  s.t. Ax ≤ b, Σx=1, x ≥ 0

通过调整 λ 控制稀疏度。
与 S-CG 对比: 精确 LP vs 近似 ADMM。
================================================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import linprog
from scipy import sparse
import time, csv
from collections import defaultdict

from fast_solver import FastSolver


# ============================================================================
# ADMM 求解器 (基于陈圣杰 2018 论文)
# ============================================================================

class ADMMSolver:
    """
    ADMM for sparse simplex LP:

    min cᵀx + λ||x||₁  s.t. Ax ≤ b, Σx=1, x ≥ 0

    增广拉格朗日:
      L = cᵀx + λ||x||₁ + (ρ/2)||Ax + s - b + u||² + (ρ/2)(Σx - 1 + v)²

    x-update: 软阈值 (soft-thresholding)
    s-update: 投影到非正锥 (s ≤ 0, 即 Ax - b ≤ 0)
    """
    def __init__(self, lam=0.01, rho=1.0, max_iter=500, tol=1e-6, verbose=False):
        self.lam = lam    # L1 penalty weight (larger = more sparse)
        self.rho = rho    # ADMM penalty parameter
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def solve(self, c, A, b):
        n = len(c); m = len(b)
        t0 = time.time()

        # Ensure dense for ADMM matrix ops
        if sparse.issparse(A):
            A_d = A.toarray()
        else:
            A_d = A

        # Initialize
        x = np.ones(n) / n          # primal: uniform
        s = np.minimum(0, b - A_d @ x)  # slack: Ax + s ≤ b
        u = np.zeros(m)             # dual for inequality
        v = 0.0                     # dual for sum constraint

        ATA = A_d.T @ A_d
        diag_ATA = np.diag(ATA)

        for it in range(self.max_iter):
            x_old = x.copy()

            # ---- x-update: soft-thresholding ----
            # min cᵀx + λ||x||₁ + (ρ/2)||Ax + s - b + u||² + (ρ/2)(Σx - 1 + v)²
            # Approximate: coordinate descent with soft-threshold
            grad = (c + self.rho * A_d.T @ (A_d @ x + s - b + u) +
                    self.rho * (np.sum(x) - 1 + v) * np.ones(n))
            # Diagonal preconditioner
            D = self.rho * (diag_ATA + 1)
            for _ in range(5):  # inner iterations
                x_new = x - grad / np.maximum(D, 1e-10)
                # Soft-threshold: prox_{λ/ρ ||·||₁}
                threshold = self.lam / self.rho
                x_new = np.sign(x_new) * np.maximum(np.abs(x_new) - threshold, 0)
                # Project to simplex
                x_new = np.maximum(x_new, 0)
                s_sum = x_new.sum()
                if s_sum > 1e-15:
                    x_new /= s_sum
                else:
                    x_new = np.ones(n) / n
                x = x_new
                grad = (c + self.rho * A_d.T @ (A_d @ x + s - b + u) +
                        self.rho * (np.sum(x) - 1 + v) * np.ones(n))

            # ---- s-update: project to s ≤ 0 ----
            s = np.minimum(0, b - A_d @ x - u)

            # ---- u, v update (dual ascent) ----
            r = A_d @ x + s - b
            u = u + r
            v = v + (np.sum(x) - 1)

            # Convergence check
            primal_res = np.linalg.norm(r) + abs(np.sum(x) - 1)
            dual_res = self.rho * np.linalg.norm(x - x_old)

            if primal_res < self.tol and dual_res < self.tol:
                break

        # Clean up
        x[x < 1e-6] = 0
        if x.sum() > 0:
            x /= x.sum()

        elapsed = time.time() - t0
        feasible = np.all(A_d @ x <= b + 1e-6)
        nnz = int(np.sum(x > 1e-6))

        return {
            'x': x, 'objective': float(c @ x),
            'time': elapsed, 'iterations': it + 1,
            'nnz': nnz, 'feasible': feasible,
            'primal_res': float(primal_res),
        }


# ============================================================================
# 对比实验
# ============================================================================

def run_comparison(n_list=[500, 1000, 2000, 5000], m=10, n_trials=5):
    """ADMM vs S-CG vs Gurobi 对比"""
    from experiments.problem_factory import generate_problem

    results = []

    for n in n_list:
        print(f"\n=== n={n} ===")

        for trial in range(n_trials):
            seed = 50000 + trial * 100
            prob = generate_problem(n=n, m=m, seed=seed, slack_tightness=0.05)
            c = prob['c']; A = prob['A_csr']; b = prob['b']

            # S-CG (精确)
            solver = FastSolver(verbose=False)
            r_scg = solver.solve(c, A, b)

            # Gurobi (精确基线)
            r_gur = {'time': np.inf, 'objective': np.inf}
            try:
                import gurobipy as gp
                t0 = time.time()
                model = gp.Model('cmp')
                model.setParam('OutputFlag', 0); model.setParam('Method', -1); model.setParam('Threads', 1)
                xv = model.addMVar(n, lb=0.0, ub=1.0)
                model.addConstr(xv.sum() == 1); model.addMConstr(A, xv, '<', b)
                model.setObjective(c @ xv, gp.GRB.MINIMIZE); model.optimize()
                r_gur['time'] = time.time() - t0
                r_gur['objective'] = model.ObjVal if model.status == gp.GRB.OPTIMAL else np.inf
            except: pass

            # ADMM (近似, 多种 λ)
            for lam in [0.001, 0.01, 0.1, 1.0]:
                admm = ADMMSolver(lam=lam, rho=1.0, max_iter=300, tol=1e-6)
                r_admm = admm.solve(c, A, b)
                gap = abs(r_admm['objective'] - r_scg['objective']) / max(abs(r_scg['objective']), 1e-15)
                results.append({
                    'n': n, 'm': m, 'trial': trial,
                    'method': f'ADMM_λ={lam}',
                    'time': r_admm['time'],
                    'objective': r_admm['objective'],
                    'gap_scg': gap * 100,
                    'nnz': r_admm['nnz'],
                    'feasible': r_admm['feasible'],
                    'iterations': r_admm['iterations'],
                })

            # S-CG
            results.append({
                'n': n, 'm': m, 'trial': trial,
                'method': 'S-CG',
                'time': r_scg['time'],
                'objective': r_scg['objective'],
                'gap_scg': 0,
                'nnz': r_scg['sparsity'],
                'feasible': True,
                'iterations': r_scg.get('iterations', 0),
            })

            # Gurobi
            if not np.isinf(r_gur['objective']):
                results.append({
                    'n': n, 'm': m, 'trial': trial,
                    'method': 'Gurobi',
                    'time': r_gur['time'],
                    'objective': r_gur['objective'],
                    'gap_scg': abs(r_gur['objective'] - r_scg['objective']) / max(abs(r_scg['objective']), 1e-15) * 100,
                    'nnz': -1, 'feasible': True, 'iterations': -1,
                })

            # Print summary
            best_admm = min([r for r in results[-6:-1] if r['method'].startswith('ADMM')],
                            key=lambda x: x['gap_scg'])
            print(f"  trial {trial}: S-CG={r_scg['time']*1000:.1f}ms "
                  f"Gurobi={r_gur['time']*1000:.1f}ms "
                  f"ADMM(best) gap={best_admm['gap_scg']:.1f}% "
                  f"nnz={best_admm['nnz']}/{r_scg['sparsity']}")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("  ADMM (Chen 2018) vs S-CG vs Gurobi")
    print("=" * 70)

    results = run_comparison(n_list=[500, 1000, 2000], n_trials=5)

    # Summary
    os.makedirs('experiments/results', exist_ok=True)
    with open('experiments/results/admm_comparison.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)

    # Summarize by method
    print("\n" + "=" * 70)
    print("  ADMM vs S-CG Summary")
    print("=" * 70)
    groups = defaultdict(list)
    for r in results: groups[r['method']].append(r)

    for method in ['S-CG', 'Gurobi'] + sorted([k for k in groups if k.startswith('ADMM')]):
        grp = groups[method]
        if grp:
            mean_t = np.mean([r['time'] for r in grp])
            mean_gap = np.mean([r['gap_scg'] for r in grp])
            mean_nnz = np.mean([r['nnz'] for r in grp])
            print(f"  {method:>12}: time={mean_t*1000:6.1f}ms "
                  f"gap={mean_gap:5.1f}% nnz={mean_nnz:.0f}")

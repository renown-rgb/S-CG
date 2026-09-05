"""
================================================================================
开源求解器对比 — S-CG vs HiGHS(dual simplex) vs HiGHS-IPM
(对应论文 tab:opensource, m=10, slack_tightness=0.1)

LP:  min c^T x  s.t.  A x <= b,  Σx = 1,  x >= 0

每配置 10 次 trial, 取中位数。seed 与 problem_grid 一致。
================================================================================
"""
import sys, os, time, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import linprog
from scipy import sparse

from problem_factory import generate_problem
from fast_solver import FastSolver

N_LIST = [10_000, 50_000]
M = 10
TRIALS = 10
OUT = os.path.join(os.path.dirname(__file__), 'results', 'opensource_compare.csv')


def seed_for(n, trial):
    return 1000 * int(np.log10(max(n, 10))) + 100 * M + trial


def solve_highs(c, A_csr, b, n, method):
    Aeq = sparse.csr_matrix(([1.0] * n, ([0] * n, range(n))), shape=(1, n))
    res = linprog(c, A_ub=A_csr, b_ub=b, A_eq=Aeq, b_eq=np.array([1.0]),
                  bounds=(0, None), method=method)
    return res


def main():
    solver = FastSolver(verbose=False)
    rows = []
    print(f'{"n":>7} {"S-CG ms":>8} {"HiGHS ms":>9} {"HiGHS-IPM ms":>12} {"IPM/SCG":>7}')
    for n in N_LIST:
        t_scg, t_hgh, t_ipm = [], [], []
        for trial in range(TRIALS):
            prob = generate_problem(n, M, seed=seed_for(n, trial), slack_tightness=0.1)
            c, A, b = prob['c'], prob['A_csr'], prob['b']

            t0 = time.perf_counter()
            solver.solve(c, A, b)
            t_scg.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            solve_highs(c, A, b, n, 'highs')
            t_hgh.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            solve_highs(c, A, b, n, 'highs-ipm')
            t_ipm.append((time.perf_counter() - t0) * 1000)

        scg, hgh, ipm = np.median(t_scg), np.median(t_hgh), np.median(t_ipm)
        rows.append({'n': n, 'scg_ms': round(float(scg), 3),
                     'highs_ms': round(float(hgh), 3),
                     'highs_ipm_ms': round(float(ipm), 3)})
        print(f'{n:>7} {scg:>8.2f} {hgh:>9.2f} {ipm:>12.2f} {ipm/scg:>6.1f}x')

    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\n已保存 -> {OUT}')


if __name__ == '__main__':
    main()

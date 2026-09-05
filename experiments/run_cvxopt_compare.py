"""
================================================================================
cvxopt 对比实验 — 重跑 (原论文 "10,000×" 声称无脚本/CSV 支撑)

LP:  min c^T x  s.t.  A x <= b,  Σx = 1,  x >= 0
     (m=10, slack_tightness=0.1, seed 与 problem_grid 一致)

cvxopt 格式:
     G = [A; -I]  (m+n) x n ,  h = [b; 0]
     A_eq = [1..1] (1 x n),  b_eq = [1]

说明:
  - cvxopt.solvers.lp 默认用稠密 LDL 分解 (KKT 规模 ~ n)。
  - n <= 10000 用稠密矩阵; n = 50000 稠密 G 会 ~20GB, 改用 cvxopt.spmatrix。
  - 时间只记录 solve 阶段 (show_progress=False)。
================================================================================
"""
import sys, os, time, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from cvxopt import matrix, spmatrix, solvers
solvers.options['show_progress'] = False
solvers.options['maxiters'] = 200

from problem_factory import generate_problem
from fast_solver import FastSolver

N_LIST = [100, 1000, 5000, 10000, 50000]
M = 10
OUT = os.path.join(os.path.dirname(__file__), 'results', 'cvxopt_compare.csv')


def seed_for(n):
    return 1000 * int(np.log10(max(n, 10))) + 100 * M + 0


def to_cvxopt_dense(c, A_dense, b, n):
    G = np.vstack([A_dense, -np.eye(n)])
    h = np.concatenate([b, np.zeros(n)])
    return (matrix(c), matrix(G), matrix(h),
            matrix(np.ones((1, n))), matrix(np.array([1.0])))


def to_cvxopt_sparse(c, A_csr, b, n):
    # G = [A; -I], h = [b; 0]
    G = spmatrix(0.0, [], [], (M + n, n))
    A_coo = A_csr.tocoo()
    for i, j, v in zip(A_coo.row, A_coo.col, A_coo.data):
        G[int(i), int(j)] = float(v)
    for i in range(n):
        G[int(M + i), int(i)] = -1.0
    h = matrix(list(b) + [0.0] * n, (M + n, 1))
    Aeq = spmatrix(1.0, [0] * n, list(range(n)), (1, n))
    beq = matrix([1.0], (1, 1))
    return (matrix(c), G, h, Aeq, beq)


def main():
    solver = FastSolver(verbose=False)
    rows = []
    print(f'{"n":>7} {"S-CG ms":>9} {"cvxopt ms":>11} {"加速":>7} {"status":>9} {"表示":>6}')
    for n in N_LIST:
        prob = generate_problem(n, M, seed=seed_for(n), slack_tightness=0.1)
        c, A_csr, b = prob['c'], prob['A_csr'], prob['b']
        A_dense = A_csr.toarray()

        # S-CG
        t0 = time.perf_counter()
        r = solver.solve(c, A_csr, b)
        t_scg = (time.perf_counter() - t0) * 1000

        # cvxopt
        dense = n <= 10000
        rep = 'dense' if dense else 'sparse'
        if dense:
            cc, G, h, Aeq, beq = to_cvxopt_dense(c, A_dense, b, n)
        else:
            cc, G, h, Aeq, beq = to_cvxopt_sparse(c, A_csr, b, n)

        t0 = time.perf_counter()
        try:
            sol = solvers.lp(cc, G, h, Aeq, beq)
            status = sol['status']
        except Exception as ex:
            status = f'ERR:{type(ex).__name__}'
            sol = None
        t_cvx = (time.perf_counter() - t0) * 1000

        spd = t_cvx / t_scg if t_scg > 0 else float('inf')
        rows.append({'n': n, 'scg_ms': round(t_scg, 3), 'cvxopt_ms': round(t_cvx, 3),
                     'speedup': round(spd, 1), 'status': status, 'repr': rep})
        print(f'{n:>7} {t_scg:>9.2f} {t_cvx:>11.1f} {spd:>6.1f}x {status:>9} {rep:>6}')

    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\n已保存 -> {OUT}')


if __name__ == '__main__':
    main()

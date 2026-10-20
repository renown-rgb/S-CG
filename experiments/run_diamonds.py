"""
================================================================================
第二个真实数据集实验 —— 钻石价格 (OpenML "diamonds", n=53,940)。

应用: 协变量平衡 / 代表性子集选择 (概率单纯形上的 LP):
    min   Σ p_i * price_i                          (最小化期望价格)
    s.t.  Σ p_i = 1,  p_i >= 0
          |Σ p_i f_i^(k) - mu_k| <= delta * |mu_k|,  k = 1..K  (匹配总体均值)

由顶点定理, 最优解至多 2K+1 = 13 个非零元, S-CG 列生成应快速求解。

对比 S-CG / HiGHS / Gurobi, N_TRIALS=30 取中位数 + IQR, 校验目标值 gap。
数据缓存到 results/diamonds.csv。
================================================================================
"""
import sys, os, time, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse

from fast_solver import FastSolver

DATA = os.path.join(os.path.dirname(__file__), 'results', 'diamonds.csv')
FEATURES = ['carat', 'depth', 'table', 'x', 'y', 'z']   # 6 个连续特征
COST = 'price'
N_TRIALS = 30


def load_diamonds():
    if os.path.exists(DATA):
        return pd.read_csv(DATA)
    from sklearn.datasets import fetch_openml
    d = fetch_openml(name='diamonds', version=1, as_frame=True, parser='auto')
    df = d.frame
    df.to_csv(DATA, index=False)
    print(f'已缓存 -> {DATA}')
    return df


def build_lp(df, delta):
    n = len(df)
    cost = df[COST].astype(float).values
    c = cost.copy()                      # min expected price
    A_rows, b_rows = [], []
    for f in FEATURES:
        fv = df[f].astype(float).values
        mu = fv.mean()
        # |Σ p f - mu| <= delta*|mu|  ->  两行
        A_rows.append(fv);          b_rows.append(mu + delta * abs(mu))    # Σ p f <= mu(1+delta)
        A_rows.append(-fv);         b_rows.append(-mu + delta * abs(mu))   # -Σ p f <= -mu(1-delta)
    A = sparse.csr_matrix(np.array(A_rows, dtype=float))
    b = np.array(b_rows, dtype=float)
    return {'c': c, 'A_csr': A, 'b': b, 'n': n, 'm': A.shape[0],
            'cost': cost, 'feat_mean': {f: float(df[f].astype(float).mean()) for f in FEATURES}}


def _med_iqr(arr):
    arr = np.asarray(arr)
    return float(np.median(arr)), float(np.percentile(arr, 75) - np.percentile(arr, 25))


def bench(prob):
    c, A, b, n = prob['c'], prob['A_csr'], prob['b'], prob['n']
    Aeq = sparse.csr_matrix(([1.0] * n, ([0] * n, range(n))), shape=(1, n))
    solver = FastSolver(verbose=False)

    r_scg = solver.solve(c, A, b)
    t_scg, t_hgh, t_gur = [], [], []
    res_h = None
    for _ in range(N_TRIALS):
        t0 = time.perf_counter(); r_scg = solver.solve(c, A, b); t_scg.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); res_h = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=np.array([1.0]), bounds=(0, None), method='highs'); t_hgh.append((time.perf_counter() - t0) * 1000)

    import gurobipy as gp
    obj_gur = np.inf
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        m = gp.Model('cb'); m.setParam('OutputFlag', 0); m.setParam('Method', -1); m.setParam('Threads', 1)
        xv = m.addMVar(n, lb=0.0, ub=1.0); m.addConstr(xv.sum() == 1); m.addMConstr(A, xv, '<', b)
        m.setObjective(c @ xv, gp.GRB.MINIMIZE); m.optimize()
        t_gur.append((time.perf_counter() - t0) * 1000)
        if m.status == gp.GRB.OPTIMAL:
            obj_gur = m.ObjVal

    scg_med, scg_iqr = _med_iqr(t_scg)
    hgh_med, hgh_iqr = _med_iqr(t_hgh)
    gur_med, gur_iqr = _med_iqr(t_gur)
    scg_obj = float(r_scg['objective'])
    highs_obj = float(res_h.fun) if res_h is not None and res_h.success else np.inf
    gap = max(abs(scg_obj - highs_obj), abs(scg_obj - obj_gur)) / max(1.0, abs(highs_obj))
    return {
        'n': n, 'm': prob['m'], 'method': r_scg['method'],
        'scg_ms': scg_med, 'scg_iqr': scg_iqr,
        'highs_ms': hgh_med, 'highs_iqr': hgh_iqr,
        'gurobi_ms': gur_med, 'gurobi_iqr': gur_iqr,
        'sp_hgh': hgh_med / scg_med, 'sp_gur': gur_med / scg_med,
        'nnz': int(np.sum(r_scg['x'] > 1e-8)), 'gap': gap,
        'scg_obj': scg_obj, 'highs_obj': highs_obj, 'gurobi_obj': obj_gur,
    }


if __name__ == '__main__':
    df = load_diamonds()
    print(f'diamonds: {df.shape[0]} 行 x {df.shape[1]} 列,  {N_TRIALS} trials (median/IQR)')
    print(f'特征均值: ' + ', '.join(f'{f}={df[f].astype(float).mean():.3f}' for f in FEATURES))
    print(f'{"delta":>6} {"m":>2} {"S-CG ms":>14} {"HiGHS ms":>15} {"Gurobi ms":>15} '
          f'{"vsH":>6} {"vsG":>6} {"nnz":>4} {"gap":>10}')
    results = []
    for delta in [0.05, 0.10, 0.20]:
        prob = build_lp(df, delta)
        r = bench(prob)
        r['delta'] = delta
        results.append(r)
        print(f"{delta:>6.2f} {r['m']:>2} {r['scg_ms']:>7.2f}±{r['scg_iqr']:.2f} "
              f"{r['highs_ms']:>7.2f}±{r['highs_iqr']:.2f} "
              f"{r['gurobi_ms']:>7.2f}±{r['gurobi_iqr']:.2f} "
              f"{r['sp_hgh']:>6.1f} {r['sp_gur']:>6.1f} {r['nnz']:>4} {r['gap']:>10.1e}")

    out = os.path.join(os.path.dirname(__file__), 'results', 'diamonds_covariate_balance.csv')
    keys = ['delta', 'n', 'm', 'method', 'scg_ms', 'scg_iqr', 'highs_ms', 'highs_iqr',
            'gurobi_ms', 'gurobi_iqr', 'sp_hgh', 'sp_gur', 'nnz', 'gap']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in keys})
    print(f'\n已保存 -> {out}')

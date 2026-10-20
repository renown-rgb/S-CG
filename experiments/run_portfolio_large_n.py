"""
================================================================================
小 m 大 n 真实组合实验 —— 放大到全 A 股 (n 最高 5295)。

数据: results/a_share_full_daily.csv (全 A 股, 61 交易日)
构建 m=5 聚合约束 LP, 对比 S-CG / HiGHS / Gurobi, 并校验目标值差距。

LP 构建完全沿用原始 groupby 管线 (load_metrics), 保证与既有结果可复现。
Sharpe 单独用日收益矩阵 (load_daily_rets) 计算, 不参与 LP 构建:
  - 组合年化收益 ann_ret_p = Σ r_j x_j
  - 组合年化波动 ann_vol_p = sqrt(x^T Σ x), Σ = 日收益样本协方差 * 252
  - Sharpe = ann_ret_p / ann_vol_p

N_TRIALS=30 取中位数 + IQR, 压降 ms 级计时噪声。
================================================================================
"""
import sys, os, time, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse

from fast_solver import FastSolver

DATA = os.path.join(os.path.dirname(__file__), 'results', 'a_share_full_daily.csv')
THETA_S = 0.05
THETA_P = 0.30
N_TRIALS = 30


def load_metrics():
    df = pd.read_csv(DATA, dtype={'trade_date': str})
    metrics = []
    for code, sdf in df.groupby('ts_code'):
        sdf = sdf.sort_values('trade_date')
        if len(sdf) < 60:
            continue
        r = sdf['close'].pct_change()
        ann_ret = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252)
        if ann_vol > 0 and not np.isnan(ann_ret):
            metrics.append({
                'ts_code': code, 'ann_return': ann_ret, 'ann_volatility': ann_vol,
                'sharpe': ann_ret / ann_vol, 'avg_amount': sdf['amount'].mean(),
            })
    return pd.DataFrame(metrics)


def load_daily_rets():
    df = pd.read_csv(DATA, dtype={'trade_date': str})
    pivot = df.pivot_table(index='trade_date', columns='ts_code', values='close',
                           aggfunc='last').sort_index()
    return pivot.pct_change(fill_method=None).dropna(how='all')   # 日收益: 日期 x 股票


def build_lp(df):
    df = df.sort_values('ann_return', ascending=False).reset_index(drop=True)
    n = len(df)
    returns = df['ann_return'].values
    volatility = df['ann_volatility'].values
    sharpe = df['sharpe'].values
    c = -returns.copy()

    A_rows, b_rows = [], []
    A_rows.append(volatility / volatility.sum()); b_rows.append(0.25)          # 风险预算
    A_rows.append(-sharpe); b_rows.append(-0.3)                                # Sharpe 质量
    top_n = max(1, int(n * 0.2))
    idx = np.argsort(df['avg_amount'].values)[-top_n:]
    row = np.zeros(n); row[idx] = 1.0; A_rows.append(row); b_rows.append(0.50) # 大盘集中度
    bot_n = max(1, int(n * 0.2))
    idx = np.argsort(df['avg_amount'].values)[:bot_n]
    row = np.zeros(n); row[idx] = 1.0; A_rows.append(-row); b_rows.append(-THETA_S)  # 小盘下限
    idx = np.argsort(returns)[-5:]
    row = np.zeros(n); row[idx] = 1.0; A_rows.append(row); b_rows.append(THETA_P)   # 前5上限

    A = sparse.csr_matrix(np.array(A_rows))
    b = np.array(b_rows)
    return {'c': c, 'A_csr': A, 'b': b, 'n': n, 'm': A.shape[0],
            'returns': returns, 'volatility': volatility, 'codes': df['ts_code'].values}


def _med_iqr(arr):
    arr = np.asarray(arr)
    return float(np.median(arr)), float(np.percentile(arr, 75) - np.percentile(arr, 25))


def bench(prob, rets_sub):
    c, A, b, n = prob['c'], prob['A_csr'], prob['b'], prob['n']
    Aeq = sparse.csr_matrix(([1.0] * n, ([0] * n, range(n))), shape=(1, n))
    solver = FastSolver(verbose=False)

    r_scg = solver.solve(c, A, b)  # 预热
    t_scg, t_hgh, t_gur = [], [], []
    res_h = None
    for _ in range(N_TRIALS):
        t0 = time.perf_counter(); r_scg = solver.solve(c, A, b); t_scg.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); res_h = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=np.array([1.0]), bounds=(0, None), method='highs'); t_hgh.append((time.perf_counter() - t0) * 1000)

    import gurobipy as gp
    obj_gur = np.inf
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        m = gp.Model('sm'); m.setParam('OutputFlag', 0); m.setParam('Method', -1); m.setParam('Threads', 1)
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

    # 组合 Sharpe 与年化收益率 (用最终 S-CG 解 x; 仅统计, 不影响计时)
    x = np.asarray(r_scg['x'], dtype=float).ravel()
    ann_ret_p = float(prob['returns'] @ x)
    Sigma = rets_sub.cov().values * 252                     # 年化协方差 (n x n, pairwise)
    var_p = float(x @ Sigma @ x)
    ann_vol_p = float(np.sqrt(max(var_p, 0.0)))
    sharpe = ann_ret_p / ann_vol_p if ann_vol_p > 0 else float('nan')   # 实现 Sharpe (含相关性)
    sharpe_w = float(x @ (prob['returns'] / prob['volatility']))        # 加权平均个股 Sharpe (忽略相关性)

    return {
        'n': n, 'm': prob['m'], 'method': r_scg['method'],
        'scg_ms': scg_med, 'scg_iqr': scg_iqr,
        'highs_ms': hgh_med, 'highs_iqr': hgh_iqr,
        'gurobi_ms': gur_med, 'gurobi_iqr': gur_iqr,
        'sp_hgh': hgh_med / scg_med, 'sp_gur': gur_med / scg_med,
        'nnz': int(np.sum(x > 1e-8)), 'sharpe': sharpe, 'sharpe_w': sharpe_w,
        'ann_return_pct': ann_ret_p * 100.0, 'gap': gap,
    }


if __name__ == '__main__':
    stock_data = load_metrics()
    rets = load_daily_rets()
    print(f'有效股票: {len(stock_data)} 只,  {N_TRIALS} trials (median/IQR)')
    print(f'{"n":>6} {"S-CG ms":>14} {"HiGHS ms":>15} {"Gurobi ms":>15} '
          f'{"vsH":>6} {"vsG":>6} {"nnz":>4} {"Sharpe":>7} {"ShW":>7} {"Ret%":>7} {"gap":>10}')
    results = []
    for n in [300, 600, 1200, 1899, 3000, 5295]:
        n = min(n, len(stock_data))
        subset = stock_data.nlargest(n, 'ann_return').reset_index(drop=True)
        prob = build_lp(subset)
        rets_sub = rets[prob['codes']]     # 列顺序与 LP 列一致
        r = bench(prob, rets_sub)
        results.append(r)
        print(f"{r['n']:>6} {r['scg_ms']:>7.2f}±{r['scg_iqr']:.2f} "
              f"{r['highs_ms']:>7.2f}±{r['highs_iqr']:.2f} "
              f"{r['gurobi_ms']:>7.2f}±{r['gurobi_iqr']:.2f} "
              f"{r['sp_hgh']:>6.1f} {r['sp_gur']:>6.1f} {r['nnz']:>4} "
              f"{r['sharpe']:>7.2f} {r['sharpe_w']:>7.2f} {r['ann_return_pct']:>7.1f} {r['gap']:>10.1e}")

    out = os.path.join(os.path.dirname(__file__), 'results', 'portfolio_large_n.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['n', 'm', 'method', 'scg_ms', 'scg_iqr',
                                          'highs_ms', 'highs_iqr', 'gurobi_ms',
                                          'gurobi_iqr', 'sp_hgh', 'sp_gur', 'nnz',
                                          'sharpe', 'sharpe_w', 'ann_return_pct', 'gap'])
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f'\n已保存 -> {out}')

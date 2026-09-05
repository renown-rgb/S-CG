"""
================================================================================
稀疏投资组合优化 — 小 m 大 n 设计 (m=5 聚合约束)

数据: experiments/results/a_share_daily.csv (2000 只 A 股, 61 个交易日)
    过滤 >=60 个交易日 → 1899 只有效股票。

构建 m=5 聚合约束的 LP:
  max  Σ r_j x_j
  s.t. Σ x_j = 1, x_j >= 0
       (1) 波动率风险预算   Σ (σ_j/Σσ_k) x_j <= 0.25
       (2) Sharpe 质量       Σ s_j x_j >= 0.3
       (3) 大盘集中度       Σ_{top20% by amount} x_j <= 0.50
       (4) 小盘下限         Σ_{bottom20% by amount} x_j >= theta_s = 0.05
       (5) 前5上限          Σ_{top5 by return} x_j <= theta_p = 0.30

求解器: S-CG / HiGHS / Gurobi, 输出与 portfolio_small_m.csv 对齐。
================================================================================
"""
import sys, os, time, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse

from fast_solver import FastSolver

DATA = os.path.join(os.path.dirname(__file__), 'results', 'a_share_daily.csv')

THETA_S = 0.05   # 小盘股下界 (Σ_{j in S} x_j >= 0.05)
THETA_P = 0.30   # 前5上界   (Σ_{j in P} x_j <= 0.30)
N_TRIALS = 7     # 时间取中位数


def load_metrics():
    df = pd.read_csv(DATA, dtype={'trade_date': str})
    metrics = []
    for code, sdf in df.groupby('ts_code'):
        sdf = sdf.sort_values('trade_date')
        if len(sdf) < 60:          # 至少 60 个交易日
            continue
        r = sdf['close'].pct_change()
        ann_ret = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252)
        if ann_vol > 0 and not np.isnan(ann_ret):
            metrics.append({
                'ts_code': code,
                'ann_return': ann_ret,
                'ann_volatility': ann_vol,
                'sharpe': ann_ret / ann_vol,
                'avg_amount': sdf['amount'].mean(),
            })
    return pd.DataFrame(metrics)


def build_small_m_lp(df):
    df = df.sort_values('ann_return', ascending=False).reset_index(drop=True)
    n = len(df)
    returns = df['ann_return'].values
    volatility = df['ann_volatility'].values
    sharpe = df['sharpe'].values
    c = -returns.copy()                      # min -r^T x  (max return)

    A_rows, b_rows = [], []

    # (1) 风险预算
    A_rows.append(volatility / volatility.sum()); b_rows.append(0.25)
    # (2) Sharpe 质量  Σ s x >= 0.3  ->  -Σ s x <= -0.3
    A_rows.append(-sharpe); b_rows.append(-0.3)
    # (3) 大盘集中度: 成交额 top 20%
    top_n = max(1, int(n * 0.2))
    idx = np.argsort(df['avg_amount'].values)[-top_n:]
    row = np.zeros(n); row[idx] = 1.0
    A_rows.append(row); b_rows.append(0.50)
    # (4) 小盘下限: 成交额 bottom 20%  >= theta_s
    bot_n = max(1, int(n * 0.2))
    idx = np.argsort(df['avg_amount'].values)[:bot_n]
    row = np.zeros(n); row[idx] = 1.0
    A_rows.append(-row); b_rows.append(-THETA_S)
    # (5) 前5上限: 收益 top 5 <= theta_p
    idx = np.argsort(returns)[-5:]
    row = np.zeros(n); row[idx] = 1.0
    A_rows.append(row); b_rows.append(THETA_P)

    A = sparse.csr_matrix(np.array(A_rows))
    b = np.array(b_rows)
    return {'c': c, 'A_csr': A, 'b': b, 'n': n, 'm': A.shape[0],
            'returns': returns, 'volatility': volatility}


def solve_scg(c, A, b):
    r = FastSolver(verbose=False).solve(c, A, b)
    return r['x'], r['objective'], r['method'], r['time']


def solve_highs(c, A, b, n):
    Aeq = sparse.csr_matrix(([1.0] * n, ([0] * n, range(n))), shape=(1, n))
    res = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=np.array([1.0]),
                  bounds=(0, None), method='highs')
    return res.x, res.fun


def solve_gurobi(c, A, b, n):
    import gurobipy as gp
    m = gp.Model("sm")
    m.setParam('OutputFlag', 0); m.setParam('Method', -1); m.setParam('Threads', 1)
    xv = m.addMVar(n, lb=0.0, ub=1.0)
    m.addConstr(xv.sum() == 1)
    m.addMConstr(A, xv, '<', b)
    m.setObjective(c @ xv, gp.GRB.MINIMIZE)
    m.optimize()
    return xv.X, m.ObjVal


def bench(prob):
    c, A, b, n = prob['c'], prob['A_csr'], prob['b'], prob['n']
    t_scg, t_hgh, t_gur = [], [], []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter(); x, *_ = solve_scg(c, A, b); t_scg.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); solve_highs(c, A, b, n); t_hgh.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); solve_gurobi(c, A, b, n); t_gur.append((time.perf_counter() - t0) * 1000)
    # 最终解(取第一次求解)
    x, obj, method, _ = solve_scg(c, A, b)
    pr = np.dot(x, prob['returns'])
    pv = np.sqrt(np.dot(x ** 2, prob['volatility'] ** 2))
    return {
        'scg_ms': float(np.median(t_scg)), 'highs_ms': float(np.median(t_hgh)),
        'gurobi_ms': float(np.median(t_gur)), 'method': method,
        'nnz': int(np.sum(x > 1e-8)), 'return': float(pr), 'sharpe': float(pr / pv),
    }


if __name__ == '__main__':
    stock_data = load_metrics()
    print(f'有效股票: {len(stock_data)} 只')
    print(f'{"n":>5} {"m":>2} {"S-CG ms":>8} {"HiGHS ms":>9} {"Gurobi ms":>9} '
          f'{"nnz":>3} {"return":>8} {"sharpe":>7}')
    results = []
    for n in [300, 600, 1200, 1899]:
        subset = stock_data.nlargest(n, 'ann_return').reset_index(drop=True)
        prob = build_small_m_lp(subset)
        r = bench(prob)
        results.append({'n': n, 'm': prob['m'], **r})
        print(f"{n:>5} {prob['m']:>2} {r['scg_ms']:>8.2f} {r['highs_ms']:>9.2f} "
              f"{r['gurobi_ms']:>9.2f} {r['nnz']:>3} {r['return']:>8.4f} {r['sharpe']:>7.2f}")

    out = os.path.join(os.path.dirname(__file__), 'results', 'portfolio_small_m.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['n', 'm', 'scg_ms', 'method', 'highs_ms',
                                          'gurobi_ms', 'nnz', 'return', 'sharpe'])
        w.writeheader()
        for r in results:
            w.writerow({'n': r['n'], 'm': r['m'], 'scg_ms': r['scg_ms'],
                        'method': r['method'], 'highs_ms': r['highs_ms'],
                        'gurobi_ms': r['gurobi_ms'], 'nnz': r['nnz'],
                        'return': r['return'], 'sharpe': r['sharpe']})
    print(f'\n已保存 -> {out}')

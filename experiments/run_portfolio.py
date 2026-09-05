"""
================================================================================
稀疏投资组合优化 — 真实A股数据 (tushare)

沪深300成分股 → 年化收益/波动率 → LP构建 → S-CG vs Gurobi vs HiGHS
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

try:
    import tushare as ts
except ImportError:
    ts = None
import pandas as pd

TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', '')

from fast_solver import FastSolver


# ============================================================================
# 1. 拉取真实数据
# ============================================================================

def fetch_csi300_data():
    """
    获取沪深300成分股近2年日线, 计算年化指标。
    优先读取本地缓存 csi300_daily_2y.csv；缺失时才联网抓取（按季度分页，避免 6000 行上限）。
    """
    cache = os.path.join(os.path.dirname(__file__), 'results', 'csi300_daily_2y.csv')

    if os.path.exists(cache):
        print(f"读取本地缓存: {cache}")
        daily = pd.read_csv(cache, dtype={'trade_date': str})
    else:
        if not TUSHARE_TOKEN or ts is None:
            raise SystemExit("未找到本地缓存 csi300_daily_2y.csv，且未设置 TUSHARE_TOKEN 环境变量或未安装 tushare。")
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()

        print("正在拉取沪深300成分股列表...")
        df_csi = pro.index_weight(index_code='000300.SH',
                                   start_date='20240101', end_date='20250601')
        latest_date = df_csi['trade_date'].max()
        stocks = df_csi[df_csi['trade_date'] == latest_date]['con_code'].tolist()
        stocks = list(set(stocks))[:500]
        print(f"  成分股数量: {len(stocks)}")

        print(f"正在拉取日线数据 ({len(stocks)}只股票, 按季度分页)...")
        ranges = [
            ('20230601', '20230930'), ('20231001', '20231231'),
            ('20240101', '20240331'), ('20240401', '20240630'),
            ('20240701', '20240930'), ('20241001', '20241231'),
            ('20250101', '20250331'), ('20250401', '20250601'),
        ]
        all_daily = []
        batch_size = 50
        for i in range(0, len(stocks), batch_size):
            batch = stocks[i:i + batch_size]
            codes = ','.join(batch)
            for s, e in ranges:
                try:
                    df = pro.daily(ts_code=codes, start_date=s, end_date=e)
                    if df is not None and len(df) > 0:
                        all_daily.append(df)
                except Exception as ex:
                    print(f"  [{codes[:20]}... {s}-{e}] ERROR: {ex}")
            print(f"  [{i+1}-{min(i+batch_size, len(stocks))}] done")

        if not all_daily:
            print("ERROR: 未获取到任何数据")
            return None

        daily = pd.concat(all_daily, ignore_index=True).drop_duplicates(subset=['ts_code', 'trade_date'])
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        daily.to_csv(cache, index=False)
        print(f"  已缓存 -> {cache}")

    print(f"  总数据: {daily.shape[0]} 行")

    # 计算每只股票的年化指标
    print("正在计算年化指标...")
    metrics = []
    stock_codes = daily['ts_code'].unique()

    for code in stock_codes:
        sdf = daily[daily['ts_code'] == code].sort_values('trade_date')

        if len(sdf) < 30:  # 至少30个交易日
            continue

        # 日收益率
        sdf['ret'] = sdf['close'].pct_change()

        # 年化收益
        ann_ret = sdf['ret'].mean() * 252
        # 年化波动
        ann_vol = sdf['ret'].std() * np.sqrt(252)
        # 总市值(最新)
        latest_close = sdf['close'].iloc[-1]
        # 平均成交额
        avg_amount = sdf['amount'].mean()

        if ann_vol > 0 and not np.isnan(ann_ret):
            metrics.append({
                'ts_code': code,
                'ann_return': ann_ret,
                'ann_volatility': ann_vol,
                'sharpe': ann_ret / ann_vol if ann_vol > 0 else 0,
                'close': latest_close,
                'avg_amount': avg_amount,
                'n_days': len(sdf),
            })

    result = pd.DataFrame(metrics)
    print(f"  有效股票: {len(result)} 只")
    print(f"  年化收益: {result['ann_return'].mean():.2%} ± {result['ann_return'].std():.2%}")
    print(f"  年化波动: {result['ann_volatility'].mean():.2%} ± {result['ann_volatility'].std():.2%}")

    return result


# ============================================================================
# 2. 投资组合 LP 构建
# ============================================================================

def build_portfolio_lp(stock_data, m_sector_constraints=True, max_weight=0.10):
    """
    构建稀疏投资组合 LP (真实数据版)

    约束 (m 个):
      1. Σx = 1
      2. x_i ≥ 0
      3. x_i ≤ max_weight (个股上限)
      4. 波动率暴露 ≤ 0.25 (risk budget)
      5. Sharpe 最低 ≥ 0.3 (质量约束)
      6. top5 集中度 ≤ 40%

    目标: min -μᵀx (最大化收益)
    """
    df = stock_data.sort_values('ann_return', ascending=False)
    n = len(df)

    # 目标: 最大化收益
    returns = df['ann_return'].values
    volatility = df['ann_volatility'].values
    sharpe = df['sharpe'].values
    c = -returns.copy()  # 最小化负收益

    constraints_A = []
    constraints_b = []

    # 约束: 个股上限
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        constraints_A.append(row)
        constraints_b.append(max_weight)

    # 约束: 波动率加权暴露 ≤ 25%
    row = volatility / volatility.sum()
    constraints_A.append(row)
    constraints_b.append(0.25)

    # 约束: Sharpe 最低 0.3 (平均Sharpe质量)
    # Σ x_i * sharpe_i ≥ 0.3 → -Σ x_i * sharpe_i ≤ -0.3
    row = -sharpe
    constraints_A.append(row)
    constraints_b.append(-0.3)

    # 约束: 大盘股集中度 (成交量最大的 top 20% ≤ 50%)
    top20_n = max(1, int(n * 0.2))
    top_vol_idx = np.argsort(df['avg_amount'].values)[-top20_n:]
    row = np.zeros(n)
    row[top_vol_idx] = 1.0
    constraints_A.append(row)
    constraints_b.append(0.50)

    # 约束: 最小多样性 (至少5个持仓, 通过前10大 ≤ 80%来实现)
    top10_n = max(1, int(n * 0.1))
    top_ret_idx = np.argsort(returns)[-top10_n:]
    row = np.zeros(n)
    row[top_ret_idx] = 1.0
    constraints_A.append(row)
    constraints_b.append(0.80)

    A = np.array(constraints_A)
    b = np.array(constraints_b)
    m = A.shape[0]

    # 稀疏存储
    A_sparse = sparse.csr_matrix(A)

    return {
        'c': c,
        'A_csr': A_sparse,
        'A_csc': A_sparse.tocsc(),
        'b': b,
        'n': n,
        'm': m,
        'returns': returns,
        'volatility': volatility,
        'sharpe': sharpe,
        'stock_codes': df['ts_code'].values,
    }


# ============================================================================
# 3. 运行实验
# ============================================================================

N_TRIALS = 7   # 时间取中位数 (对齐 run_portfolio_small_m.py)


def run_comparison(prob, label):
    """N_TRIALS 次求解取中位数计时, 消除单次运行噪声。"""
    c = prob['c']; A = prob['A_csr']; b = prob['b']; n = prob['n']
    A_eq = sparse.csr_matrix(([1.0]*n, ([0]*n, range(n))), shape=(1, n))

    solver = FastSolver(verbose=False)

    # S-CG: 预热一次(丢弃)后计时 N_TRIALS 次, 取中位数
    r_scg = solver.solve(c, A, b)
    t_scg = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        r_scg = solver.solve(c, A, b)
        t_scg.append((time.perf_counter() - t0) * 1000)
    scg_ms = float(np.median(t_scg))

    # HiGHS
    t_hgh = []
    res_h = None
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        res_h = linprog(c, A_ub=A, b_ub=b,
                        A_eq=A_eq, b_eq=np.array([1.0]),
                        bounds=(0, None), method='highs')
        t_hgh.append((time.perf_counter() - t0) * 1000)
    hgh_ms = float(np.median(t_hgh))

    # Gurobi
    t_gur = np.inf; obj_gur = np.inf; nnz_gur = -1
    try:
        import gurobipy as gp
        t_gur_li = []
        for _ in range(N_TRIALS):
            t0 = time.perf_counter()
            model = gp.Model("pf")
            model.setParam('OutputFlag', 0)
            model.setParam('Method', -1)
            model.setParam('Threads', 1)
            xv = model.addMVar(n, lb=0.0, ub=1.0)
            model.addConstr(xv.sum() == 1)
            model.addMConstr(A, xv, '<', b)
            model.setObjective(c @ xv, gp.GRB.MINIMIZE)
            model.optimize()
            t_gur_li.append((time.perf_counter() - t0) * 1000)
            if model.status == gp.GRB.OPTIMAL:
                obj_gur = model.ObjVal
                nnz_gur = int(np.sum(xv.X > 1e-8))
        t_gur = float(np.median(t_gur_li)) / 1000.0
    except ImportError:
        pass

    # 投资组合指标(用最后一次 S-CG 解)
    scg_x = r_scg['x']
    port_ret = np.dot(scg_x, prob['returns'])
    port_vol = np.sqrt(np.dot(scg_x**2, prob['volatility']**2))
    port_sharpe = port_ret / port_vol if port_vol > 0 else 0
    nnz = int(np.sum(scg_x > 1e-8))

    # Top holdings
    top_idx = np.argsort(scg_x)[-5:][::-1]
    top_holdings = [(prob['stock_codes'][i], float(scg_x[i])) for i in top_idx
                    if scg_x[i] > 1e-4]

    return {
        'label': label,
        'n': n, 'm': prob['m'],
        'scg_time': scg_ms / 1000.0,
        'scg_obj': r_scg['objective'],
        'scg_nnz': nnz,
        'scg_method': r_scg['method'],
        'highs_time': hgh_ms / 1000.0,
        'highs_obj': float(res_h.fun) if res_h.success else np.inf,
        'gurobi_time': t_gur,
        'gurobi_obj': obj_gur,
        'gurobi_nnz': nnz_gur,
        'port_return': float(port_ret),
        'port_volatility': float(port_vol),
        'port_sharpe': float(port_sharpe),
        'top_holdings': top_holdings,
    }


if __name__ == "__main__":
    import pandas as pd

    print("=" * 70)
    print("  稀疏投资组合优化 — 真实A股数据 (tushare)")
    print("=" * 70)

    # 获取数据
    stock_data = fetch_csi300_data()

    if stock_data is None or len(stock_data) < 30:
        print("数据不足，使用模拟数据回退...")
        from run_portfolio_synthetic import generate_stock_universe
        # Just use what we have
        raise SystemExit("请检查 tushare token 或网络连接")

    print(f"\n有效股票: {len(stock_data)} 只")
    print(f"收益范围: {stock_data['ann_return'].min():.2%} ~ {stock_data['ann_return'].max():.2%}")
    print(f"波动范围: {stock_data['ann_volatility'].min():.2%} ~ {stock_data['ann_volatility'].max():.2%}")

    # 不同规模的实验
    configs = [
        (100, "沪深300前100 (按收益)"),
        (200, "沪深300前200"),
        (300, "沪深300 (全部有效)"),
    ]

    results = []
    for n_stocks, label in configs:
        n_stocks = min(n_stocks, len(stock_data))
        # 取收益最高的 n_stocks 只股票
        subset = stock_data.nlargest(n_stocks, 'ann_return').reset_index(drop=True)

        print(f"\n--- {label} ({n_stocks}只) ---")

        prob = build_portfolio_lp(subset)
        r = run_comparison(prob, label)
        results.append(r)

        sp_gh = r['highs_time'] / r['scg_time'] if r['scg_time'] > 0 else 0
        sp_gr = r['gurobi_time'] / r['scg_time'] if r['scg_time'] > 0 else 0
        print(f"  S-CG: {r['scg_time']:.4f}s [{r['scg_method']}] "
              f"HiGHS: {r['highs_time']:.4f}s Gurobi: {r['gurobi_time']:.4f}s")
        print(f"  加速: vsHiGHS={sp_gh:.1f}x vsGurobi={sp_gr:.1f}x")
        print(f"  组合: ret={r['port_return']:.2%} vol={r['port_volatility']:.2%} "
              f"Sharpe={r['port_sharpe']:.2f} 持仓={r['scg_nnz']}只")
        print(f"  重仓: {', '.join([f'{c}({w:.1%})' for c,w in r['top_holdings'][:5]])}")

    # 保存结果
    os.makedirs('experiments/results', exist_ok=True)
    save_results = [{k: v for k, v in r.items() if k != 'top_holdings'}
                    for r in results]
    with open('experiments/results/portfolio_real.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=save_results[0].keys())
        w.writeheader()
        w.writerows(save_results)

    print(f"\n结果已保存到 experiments/results/portfolio_real.csv")

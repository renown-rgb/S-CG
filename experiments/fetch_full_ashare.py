"""
================================================================================
拉取全 A 股日线数据（全部上市股票），用于把真实组合实验放大到 n >= 5000。

复用 run_portfolio_small_m.py 的 61 个交易日窗口 (2025-03-03 .. 2025-05-30)，
按交易日逐日调用 pro.daily(trade_date=...) 拉取全市场，缓存到
results/a_share_full_daily.csv。
================================================================================
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import tushare as ts

TOKEN = os.environ.get('TUSHARE_TOKEN', '')
DATA = os.path.join(os.path.dirname(__file__), 'results', 'a_share_daily.csv')
OUT = os.path.join(os.path.dirname(__file__), 'results', 'a_share_full_daily.csv')


def main():
    if not TOKEN:
        raise SystemExit('TUSHARE_TOKEN 环境变量未设置')
    ts.set_token(TOKEN)
    pro = ts.pro_api()

    # 复用已有文件的 61 个交易日窗口，保持一致
    existing = pd.read_csv(DATA, dtype={'trade_date': str})
    dates = sorted(existing['trade_date'].unique())
    print(f'{len(dates)} 个交易日: {dates[0]} .. {dates[-1]}')

    frames = []
    for i, d in enumerate(dates):
        ok = False
        for attempt in range(3):
            try:
                df = pro.daily(trade_date=d)
                if df is not None and len(df):
                    frames.append(df)
                ok = True
                break
            except Exception as e:
                print(f'  [{d}] 第 {attempt + 1} 次失败: {e}')
                time.sleep(1.5)
        if not ok:
            print(f'  [{d}] 三次均失败，跳过')
        if (i + 1) % 10 == 0:
            print(f'  {i + 1}/{len(dates)} 天完成')
        time.sleep(0.35)

    if not frames:
        raise SystemExit('未获取到任何数据')
    full = pd.concat(frames, ignore_index=True).drop_duplicates(subset=['ts_code', 'trade_date'])
    full = full.sort_values(['ts_code', 'trade_date'])
    full.to_csv(OUT, index=False)
    print(f'已保存 -> {OUT}: {full.shape[0]} 行, {full["ts_code"].nunique()} 只股票')


if __name__ == '__main__':
    main()

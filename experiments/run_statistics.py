"""
================================================================================
统计显著性检验 — S-CG vs Gurobi / HiGHS

数据源: experiments/results/scalability_with_gurobi.csv
        (30 次/配置 = 10 trial × 3 tightness {0.01, 0.05, 0.1})

检验:
  - Wilcoxon signed-rank 检验 (配对, 同一实例分别由两个求解器求解)
  - Cohen's d (合并标准差, 总体方差 ddof=0):
        d = mean(Gurobi - S-CG) / sqrt((var_G + var_S)/2)
    符号约定: d > 0 表示 S-CG 更快; d < 0 表示对方求解器更快。

说明: p 值下界 1.86e-9 = 2 / 2^30, 是 30 个配对样本 Wilcoxon 精确检验的最小值,
      表示 S-CG 在全部 30 次试验中均快于对方。
================================================================================
"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

SRC = os.path.join(os.path.dirname(__file__), 'results', 'scalability_with_gurobi.csv')
OUT = os.path.join(os.path.dirname(__file__), 'results', 'statistics.csv')

# 论文 tab:statistics 的 (n, m) 行
ROWS = [(100, 10), (1000, 20), (1000, 50), (10000, 20),
        (100000, 10), (500000, 10), (1000000, 10)]


def wilcoxon_p(a, b):
    """配对 Wilcoxon signed-rank 检验的 p 值 (双侧)。"""
    return float(stats.wilcoxon(a, b).pvalue)


def cohen_d(a, b):
    """Cohen's d: 正=对方更慢(我方更快); 负=对方更快。ddof=0 (总体方差)。"""
    diff = b - a                     # 对方时间 - 我方时间
    return float(diff.mean() / np.sqrt((b.var(ddof=0) + a.var(ddof=0)) / 2))


def main():
    df = pd.read_csv(SRC)
    rows = []
    print(f'{"(n,m)":>11} {"n行数":>5} {"p vs Gurobi":>12} {"p vs HiGHS":>12} '
          f'{"d vs Gurobi":>11} {"d vs HiGHS":>11} {"方向":>8}')
    for n, m in ROWS:
        sub = df[(df.n == n) & (df.m == m)]
        if len(sub) == 0:
            print(f'({n},{m}) 无数据!')
            continue
        scg = sub['s-cg_time'].values
        gur = sub['gurobi_time'].values
        hgh = sub['highs_time'].values
        pg, ph = wilcoxon_p(scg, gur), wilcoxon_p(scg, hgh)
        dg, dh = cohen_d(scg, gur), cohen_d(scg, hgh)
        # 方向: d>0 -> S-CG 更快
        dir_g = 'S-CG快' if dg > 0 else ('Gurobi快' if dg < 0 else '平')
        dir_h = 'S-CG快' if dh > 0 else ('HiGHS快' if dh < 0 else '平')
        rows.append({'n': n, 'm': m, 'p_gurobi': pg, 'p_highs': ph,
                     'cohen_d_gurobi': dg, 'cohen_d_highs': dh,
                     'dir_gurobi': dir_g, 'dir_highs': dir_h})
        print(f'{(n,m)!s:>11} {len(sub):>5} {pg:>12.2e} {ph:>12.2e} '
              f'{dg:>11.2f} {dh:>11.2f} {dir_g:>8}/{dir_h}')

    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\n已保存 -> {OUT}')


if __name__ == '__main__':
    main()

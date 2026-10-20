"""
================================================================================
论文图表生成 — 论文收录的 2 张图（图2、图5）

依赖: matplotlib
数据来源: experiments/results/*.csv
================================================================================
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv
import os
from collections import defaultdict

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

OUTPUT_DIR = 'paper/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette
COLORS = {
    'S-CG': '#2196F3',
    'Gurobi': '#F44336',
    'HiGHS': '#4CAF50',
}


def load_csv(filepath):
    """加载CSV结果"""
    results = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 转换数值字段
            for k, v in row.items():
                try:
                    row[k] = float(v)
                except (ValueError, TypeError):
                    pass
            results.append(row)
    return results


def group_by(results, *keys):
    """按指定键分组"""
    groups = defaultdict(list)
    for r in results:
        gk = tuple(r.get(k) for k in keys)
        groups[gk].append(r)
    return groups


# ============================================================================
# Fig 2: 时间 vs n (log-log)
# ============================================================================

def fig2_time_vs_n(results_file='experiments/results/scalability_final.csv'):
    """时间 vs n 的 log-log 图 (含误差带)"""
    results = load_csv(results_file)
    if not results:
        print(f"No results found in {results_file}, skipping Fig 2")
        return

    # Filter for m=10, tightness=0.1 (median case)
    filtered = [r for r in results
                if r.get('m') == 10 and abs(r.get('tightness', 0) - 0.1) < 0.01]

    if not filtered:
        # Fallback: use all
        filtered = results

    # Group by n
    by_n = group_by(filtered, 'n')
    n_values = sorted(set(r['n'] for r in filtered))

    fig, ax = plt.subplots(figsize=(8, 6))

    solver_keys = {
        'S-CG': ('s-cg_time', 's-cg_obj'),
        'Gurobi': ('gurobi_time', 'gurobi_obj'),
        'HiGHS': ('highs_time', 'highs_obj'),
    }

    for sname, (time_key, _obj_key) in solver_keys.items():
        means = []; stds = []; ns = []
        for n in n_values:
            group = by_n.get((n,), [])
            times = [r[time_key] for r in group
                     if time_key in r and not np.isinf(float(r[time_key]))]
            if times:
                ns.append(n)
                means.append(np.mean(times))
                stds.append(np.std(times))

        if ns:
            ax.errorbar(ns, means, yerr=stds,
                        marker='o', markersize=6, linewidth=2,
                        capsize=3, label=sname, color=COLORS.get(sname),
                        markeredgewidth=1, markeredgecolor='white')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Number of variables $n$')
    ax.set_ylabel('Wall-clock time (s)')
    ax.set_title('S-CG vs scipy HiGHS: Solution Time vs Problem Size')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f'{int(x):,}'))

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig2_time_vs_n.pdf')
    fig.savefig(f'{OUTPUT_DIR}/fig2_time_vs_n.png')
    plt.close()
    print(f"  Fig 2 saved")


# ============================================================================
# Fig 5: 消融柱状图
# ============================================================================

def fig5_ablation(results_file='experiments/results/ablation.csv'):
    """消融实验柱状图"""
    results = load_csv(results_file)
    if not results:
        print(f"No results found, skipping Fig 5")
        return

    variants = ['S-CG_full', 'S-CG_noL1', 'S-CG_dense', 'HiGHS_direct']
    labels = ['Full S-CG', 'Without Stage 1', 'Dense Pricing', 'HiGHS Direct Solution']

    by_var = group_by(results, 'n', 'variant')
    n_vals = sorted(set(r['n'] for r in results))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(n_vals))
    width = 0.2
    colors = ['#2196F3', '#FF9800', '#F44336', '#9E9E9E']

    for i, (variant, label) in enumerate(zip(variants, labels)):
        means = []
        for n in n_vals:
            group = by_var.get((n, variant), [])
            times = [r['time'] for r in group
                     if not np.isinf(float(r.get('time', np.inf)))]
            means.append(np.mean(times) * 1000 if times else 0)  # to ms

        ax.bar(x + i * width, means, width, label=label,
               color=colors[i], edgecolor='white', linewidth=0.5)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'{int(n):,}' for n in n_vals])
    ax.set_xlabel('Number of variables $n$')
    ax.set_ylabel('Mean time (ms)')
    ax.set_title('Ablation Study: Component Contributions')
    ax.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig5_ablation.pdf')
    fig.savefig(f'{OUTPUT_DIR}/fig5_ablation.png')
    plt.close()
    print(f"  Fig 5 saved")


if __name__ == "__main__":
    print("Generating all figures...")
    fig2_time_vs_n()
    fig5_ablation()
    print(f"\nAll figures saved to {OUTPUT_DIR}/")

"""
================================================================================
论文图表生成 — 全部6张图

依赖: matplotlib, seaborn
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
    '01GA': '#FF9800',
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
# Fig 3: 加速比 vs n
# ============================================================================

def fig3_speedup(results_file='experiments/results/scalability_final.csv'):
    """加速比 vs n (不同 m)"""
    results = load_csv(results_file)
    if not results:
        print(f"No results found, skipping Fig 3")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    m_values = sorted(set(r.get('m', 10) for r in results))
    for m in m_values:
        filtered = [r for r in results
                    if r.get('m') == m and abs(r.get('tightness', 0) - 0.1) < 0.01]
        by_n = group_by(filtered, 'n')
        n_vals = sorted(set(r['n'] for r in filtered))

        speedups_gur = []; speedups_hgh = []
        ns_plot = []
        for n in n_vals:
            group = by_n.get((n,), [])
            scg_t = [r['s-cg_time'] for r in group
                     if 's-cg_time' in r and not np.isinf(float(r['s-cg_time']))]
            gur_t = [r['gurobi_time'] for r in group
                     if 'gurobi_time' in r and not np.isinf(float(r.get('gurobi_time', np.inf)))]
            hgh_t = [r['highs_time'] for r in group
                     if 'highs_time' in r and not np.isinf(float(r.get('highs_time', np.inf)))]
            if scg_t and gur_t:
                speedups_gur.append(np.mean(gur_t) / np.mean(scg_t))
                speedups_hgh.append(np.mean(hgh_t) / np.mean(scg_t) if hgh_t else 0)
                ns_plot.append(n)

        if ns_plot:
            ax.plot(ns_plot, speedups_gur,
                    marker='o', markersize=6, linewidth=2,
                    label=f'$m={int(m)}$ vs Gurobi', color='#F44336')
            ax.plot(ns_plot, speedups_hgh,
                    marker='s', markersize=5, linewidth=1.5, linestyle='--',
                    label=f'$m={int(m)}$ vs HiGHS', color='#4CAF50')

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Break-even')
    ax.set_xscale('log')
    ax.set_xlabel('Number of variables $n$')
    ax.set_ylabel('Speedup $T_{baseline} / T_{S-CG}$')
    ax.set_title('Speedup of S-CG over Gurobi and scipy HiGHS')
    ax.legend(loc='upper left', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f'{int(x):,}'))

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig3_speedup.pdf')
    fig.savefig(f'{OUTPUT_DIR}/fig3_speedup.png')
    plt.close()
    print(f"  Fig 3 saved")


# ============================================================================
# Fig 4: Level 分布
# ============================================================================

def fig4_method_breakdown(results_file='experiments/results/scalability_final.csv'):
    """各Level解决比例 (柱状图)"""
    results = load_csv(results_file)
    if not results:
        print(f"No results found, skipping Fig 4")
        return

    by_n = group_by(results, 'n')
    n_vals = sorted(set(r['n'] for r in results))

    methods = ['single_column_verified', 'column_generation']
    method_labels = ['Level 1 (KKT Cert)', 'Level 2 (ColGen)']

    data = {m: [] for m in methods}
    for n in n_vals:
        group = by_n.get((n,), [])
        total = len(group)
        for method in methods:
            count = sum(1 for r in group
                        if r.get('s-cg_method', '') == method)
            data[method].append(count / total * 100 if total else 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(n_vals))
    width = 0.2
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    for i, (method, label) in enumerate(zip(methods, method_labels)):
        ax.bar(x + i * width, data[method], width, label=label,
               color=colors[i], edgecolor='white', linewidth=0.5)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'{int(n):,}' for n in n_vals])
    ax.set_xlabel('Number of variables $n$')
    ax.set_ylabel('Percentage of instances (%)')
    ax.set_title('Method Distribution: Which Level Solved the Problem')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig4_method_distribution.pdf')
    fig.savefig(f'{OUTPUT_DIR}/fig4_method_distribution.png')
    plt.close()
    print(f"  Fig 4 saved")


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


# ============================================================================
# Fig 6: 退化相变
# ============================================================================

def fig6_degeneracy(results_file='experiments/results/degeneracy.csv'):
    """退化相变图"""
    results = load_csv(results_file)
    if not results:
        print(f"No results found, skipping Fig 6")
        return

    by_n = group_by(results, 'n')
    n_vals = sorted(set(r['n'] for r in results))

    fig, ax = plt.subplots(figsize=(8, 6))

    for n in n_vals:
        group = by_n.get((n,), [])
        tightness = [r['tightness'] for r in group]
        success = [r['l1_success_rate'] * 100 for r in group]

        # Sort by tightness
        pairs = sorted(zip(tightness, success))
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                marker='o', markersize=5, linewidth=2,
                label=f'$n={int(n):,}$')

    ax.set_xscale('log')
    ax.set_xlabel('Constraint tightness $\\tau$')
    ax.set_ylabel('Level 1 success rate (%)')
    ax.set_title('Degeneracy Phase Transition')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig6_degeneracy.pdf')
    fig.savefig(f'{OUTPUT_DIR}/fig6_degeneracy.png')
    plt.close()
    print(f"  Fig 6 saved")


# ============================================================================
# 批量生成
# ============================================================================

def fig7_complexity_boundary():
    """复杂度边界可视化 — P/NP-hard 分界线"""
    import numpy as np
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ms = np.arange(1, 101)
    ks = np.arange(0, 101)

    # === Left: S-SLP (Simplex) ===
    K, M_grid = np.meshgrid(ks, ms)
    region_p = K >= M_grid + 1
    p_deg = (1 - 0.005)**M_grid
    region_d = (p_deg >= 0.95) & ~region_p
    Z = np.zeros_like(K, dtype=float)
    Z[region_p] = 2.0
    Z[region_d] = 1.0

    cmap = plt.cm.colors.ListedColormap(['#EF9A9A', '#A5D6A7', '#4CAF50'])
    ax1.pcolormesh(K, M_grid, Z, cmap=cmap, vmin=0, vmax=2, shading='auto')
    ax1.plot(ks, ks + 1, 'k--', linewidth=2, label='$K=m+1$')
    ax1.set_xlabel('$K$'); ax1.set_ylabel('$m$')
    ax1.set_title('S-SLP: simplex constraints')
    ax1.set_xlim(0, 100); ax1.set_ylim(0, 100)
    ax1.legend(loc='lower right')
    ax1.text(60, 15, 'P', fontsize=18, ha='center', fontweight='bold', color='white')
    ax1.text(15, 60, 'Degenerate\n>= 95%', fontsize=10, ha='center', color='#1B5E20')
    ax1.text(8, 92, 'NP-hard', fontsize=9, ha='center', color='#B71C1C')

    # === Right: G-SLP (General) ===
    Z2 = np.zeros_like(K, dtype=float)
    Z2[K >= M_grid] = 2.0
    ax2.pcolormesh(K, M_grid, Z2, cmap=cmap, vmin=0, vmax=2, shading='auto')
    ax2.plot(ks, ks, 'k--', linewidth=2, label='$K=m$')
    ax2.set_xlabel('$K$'); ax2.set_ylabel('$m$')
    ax2.set_title('G-SLP: $x$ unrestricted')
    ax2.set_xlim(0, 100); ax2.set_ylim(0, 100)
    ax2.legend(loc='lower right')
    ax2.text(65, 15, 'P', fontsize=18, ha='center', fontweight='bold', color='white')
    ax2.text(15, 75, 'NP-hard\n(C(m,K) search)', fontsize=10, ha='center', color='#B71C1C')

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(f'{OUTPUT_DIR}/fig_complexity_boundary.{ext}')
    plt.close()
    print(f'  Fig 7 saved')


if __name__ == "__main__":
    print("Generating all figures...")
    fig2_time_vs_n()
    fig3_speedup()
    fig4_method_breakdown()
    fig5_ablation()
    fig6_degeneracy()
    fig7_complexity_boundary()
    print(f"\nAll figures saved to {OUTPUT_DIR}/")

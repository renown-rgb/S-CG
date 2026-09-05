"""
================================================================================
Performance Profile (Dolan-Moré) — 优化论文标准图

显示 S-CG vs Gurobi vs HiGHS 的性能分布。
横轴 τ = 性能比率, 纵轴 P(ratio ≤ τ) = 累积概率。
靠左上角的曲线 = 更好的求解器。
================================================================================
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv, os

OUTPUT_DIR = 'paper/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({'font.size': 12, 'font.family': 'serif',
                      'figure.dpi': 150, 'savefig.dpi': 150})


def load_csv(fp):
    results = []
    with open(fp) as f:
        for row in csv.DictReader(f):
            for k, v in list(row.items()):
                try: row[k] = float(v)
                except: pass
            results.append(row)
    return results


def performance_profile(solver_times, solver_names, title, filename):
    """
    solver_times: dict {solver_name: [times_per_instance]}
    """
    n_instances = len(next(iter(solver_times.values())))
    tau_max = 100  # max ratio to plot

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'S-CG': '#2196F3', 'Gurobi': '#F44336', 'HiGHS': '#4CAF50'}

    for solver in solver_names:
        times = np.array(solver_times[solver])
        # For each instance, ratio = time / min_time
        min_times = np.min([solver_times[s] for s in solver_names], axis=0)
        ratios = times / np.maximum(min_times, 1e-15)

        # Performance profile: P(ratio ≤ τ)
        tau_vals = np.logspace(-1, np.log10(tau_max), 200)
        probs = [np.mean(ratios <= tau) for tau in tau_vals]

        ax.step(tau_vals, probs, where='post', linewidth=2,
                label=solver, color=colors.get(solver, 'gray'))

    ax.set_xscale('log')
    ax.set_xlabel(r'Performance ratio $\tau$')
    ax.set_ylabel(r'$P(T \leq \tau \cdot T_{\min})$')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.8, tau_max)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/{filename}.pdf')
    fig.savefig(f'{OUTPUT_DIR}/{filename}.png')
    plt.close()
    print(f'  {filename} saved')


if __name__ == "__main__":
    # Load scalability data
    results = load_csv('experiments/results/scalability_with_gurobi.csv')
    if not results:
        print("No data found, skipping")
        exit(0)

    # Collect per-solver times over all valid results
    solver_times = {'S-CG': [], 'Gurobi': [], 'HiGHS': []}
    for r in results:
        t_scg = r.get('s-cg_time', np.inf)
        t_gur = r.get('gurobi_time', np.inf)
        t_hgh = r.get('highs_time', np.inf)
        # Only include if all three solvers returned valid times
        if (not np.isinf(t_scg) and t_scg > 0 and
            not np.isinf(t_gur) and t_gur > 0 and
            not np.isinf(t_hgh) and t_hgh > 0):
            solver_times['S-CG'].append(t_scg)
            solver_times['Gurobi'].append(t_gur)
            solver_times['HiGHS'].append(t_hgh)

    n_valid = len(solver_times['S-CG'])
    print(f"Valid instances (all 3 solvers): {n_valid}")

    if n_valid > 0:
        # Overall profile
        performance_profile(solver_times, ['S-CG', 'Gurobi', 'HiGHS'],
                            'Performance Profile (All instances)',
                            'fig_perf_profile_all')

        # Large n only
        large_times = {'S-CG': [], 'Gurobi': [], 'HiGHS': []}
        for i, r in enumerate(results):
            if r.get('n', 0) >= 100000:
                t_scg = r.get('s-cg_time', np.inf)
                t_gur = r.get('gurobi_time', np.inf)
                t_hgh = r.get('highs_time', np.inf)
                if (not np.isinf(t_scg) and t_scg > 0 and
                    not np.isinf(t_gur) and t_gur > 0 and
                    not np.isinf(t_hgh) and t_hgh > 0):
                    large_times['S-CG'].append(t_scg)
                    large_times['Gurobi'].append(t_gur)
                    large_times['HiGHS'].append(t_hgh)

        if large_times['S-CG']:
            performance_profile(large_times, ['S-CG', 'Gurobi', 'HiGHS'],
                                r'Performance Profile ($n \geq 10^5$)',
                                'fig_perf_profile_large_n')

    print("Performance profiles done.")

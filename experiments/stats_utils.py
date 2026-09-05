"""
================================================================================
统计模块 — 实验数据分析 + LaTeX 表格生成
================================================================================
"""
import numpy as np
from collections import defaultdict
from typing import List, Dict
import csv
import json


def group_by(results: List[Dict], *keys) -> Dict:
    """按指定键分组"""
    groups = defaultdict(list)
    for r in results:
        gk = tuple(r.get(k) for k in keys)
        groups[gk].append(r)
    return dict(groups)


def summarize(results: List[Dict]):
    """打印实验摘要"""
    if not results:
        print("No results to summarize.")
        return

    solvers = ['s-cg', 'gurobi', 'highs', '01ga']
    metrics = ['obj', 'time', 'gap', 'nnz']

    # 按 n 分组
    by_n = group_by(results, 'n')
    for n in sorted(by_n.keys()):
        group = by_n[n]
        n_results = len(group)
        difficulties = defaultdict(int)
        for r in group:
            difficulties[r.get('difficulty', 'unknown')] += 1

        print(f"\n--- n={n} ({n_results} trials) ---")
        print(f"  Difficulty: {dict(difficulties)}")

        for solver in solvers:
            obj_key = solver + '_obj'
            time_key = solver + '_time'
            gap_key = solver + '_gap'

            objs = [r[obj_key] for r in group
                    if obj_key in r and not np.isinf(r[obj_key])]
            times = [r[time_key] for r in group
                     if time_key in r and not np.isinf(r[time_key])]
            gaps = [r[gap_key] for r in group
                    if gap_key in r and not np.isinf(r[gap_key])]

            if objs:
                print(f"  {solver:>6}: obj={np.mean(objs):.6f}±{np.std(objs):.6f}  "
                      f"time={np.mean(times):.4f}s±{np.std(times):.4f}  "
                      f"gap={np.mean(gaps):.2e}%±{np.std(gaps):.2e}  "
                      f"n_valid={len(objs)}/{n_results}")


def latex_table_scalability(results: List[Dict],
                             solvers=['S-CG', 'Gurobi', 'HiGHS'],
                             filename: str = None) -> str:
    """生成可扩展性 LaTeX 表格"""
    groups = group_by(results, 'n', 'm')

    header = (r"\begin{table}[htbp]" + "\n"
              r"\centering" + "\n"
              r"\caption{Scalability comparison: S-CG vs Gurobi vs scipy HiGHS. "
              r"Mean $\pm$ std over 10 trials.}" + "\n"
              r"\label{tab:scalability}" + "\n"
              r"\begin{tabular}{rr" + "r" * (len(solvers) * 2) + "}" + "\n"
              r"\hline" + "\n")

    col_header = r"$n$ & $m$ "
    for s in solvers:
        col_header += f"& {s} (s) & {s} obj "
    col_header += r"\\ \hline"

    rows = [header, col_header]

    for key in sorted(groups.keys()):
        group = groups[key]
        n, m = key
        row = f"{n:,} & {m} "

        for s in solvers:
            sk = s.lower().replace('-', '')
            times = [r[sk + '_time'] for r in group
                     if sk + '_time' in r and not np.isinf(r[sk + '_time'])]
            objs = [r[sk + '_obj'] for r in group
                    if sk + '_obj' in r and not np.isinf(r[sk + '_obj'])]

            if times:
                row += f" & {np.mean(times):.3f}$\\pm${np.std(times):.2f}"
            else:
                row += " & ---"

            if objs:
                ref = group[0].get('reference_obj', 1.0)
                gap = abs(np.mean(objs) - ref) / max(abs(ref), 1e-15) * 100
                row += f" & {np.mean(objs):.6f}"
            else:
                row += " & ---"

        row += r" \\"
        rows.append(row)

    footer = (r"\hline" + "\n"
              r"\end{tabular}" + "\n"
              r"\end{table}")

    table = "\n".join(rows + [footer])

    if filename:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(table)
        print(f"LaTeX table saved to {filename}")

    return table


def speedup_summary(results: List[Dict]) -> Dict:
    """计算加速比摘要"""
    groups = group_by(results, 'n')

    summary = {}
    for n, group in groups.items():
        scg_times = [r['s-cg_time'] for r in group
                     if 's-cg_time' in r and not np.isinf(r['s-cg_time'])]
        gur_times = [r['gurobi_time'] for r in group
                     if 'gurobi_time' in r and not np.isinf(r['gurobi_time'])]
        hgh_times = [r['highs_time'] for r in group
                     if 'highs_time' in r and not np.isinf(r['highs_time'])]

        summary[n] = {
            's-cg_mean': np.mean(scg_times) if scg_times else np.inf,
            'gurobi_mean': np.mean(gur_times) if gur_times else np.inf,
            'highs_mean': np.mean(hgh_times) if hgh_times else np.inf,
            'speedup_vs_gurobi': (np.mean(gur_times) / np.mean(scg_times)
                                  if scg_times and gur_times else np.inf),
            'speedup_vs_highs': (np.mean(hgh_times) / np.mean(scg_times)
                                 if scg_times and hgh_times else np.inf),
            'n_trials': len(group),
        }
    return summary


def method_breakdown(results: List[Dict]) -> Dict:
    """统计各Level解决比例"""
    methods = defaultdict(int)
    for r in results:
        m = r.get('s-cg_method', 'unknown')
        methods[m] += 1

    total = sum(methods.values())
    return {k: {'count': v, 'pct': v/total*100 if total else 0}
            for k, v in methods.items()}

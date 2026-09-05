"""
================================================================================
统一实验框架 — S-CG vs Gurobi vs scipy HiGHS vs 0-1GA

用法:
  python experiment_runner.py --quick       # 快速测试 (1 trial)
  python experiment_runner.py --full        # 完整实验 (10 trials)
  python experiment_runner.py --n 100000    # 仅测试特定n
================================================================================
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import linprog
from scipy import sparse
import time
import json
import csv
import argparse
from typing import Dict, List, Optional

from fast_solver import FastSolver
from problem_factory import generate_problem, problem_grid


# ============================================================================
# 求解器接口
# ============================================================================

def solve_scg(prob: Dict) -> Dict:
    """S-CG (我们的方法)"""
    solver = FastSolver(verbose=False)
    result = solver.solve(prob['c'], prob['A_csr'], prob['b'])
    return {
        'solver': 'S-CG',
        'objective': result['objective'],
        'time': result['time'],
        'sparsity': result['sparsity'],
        'method': result['method'],
        'iterations': result.get('iterations', 0),
    }


def solve_gurobi(prob: Dict, method=-1, threads=1) -> Dict:
    """Gurobi (公平对比)"""
    try:
        import gurobipy as gp
    except ImportError:
        return {'solver': 'Gurobi', 'objective': np.inf, 'time': np.inf,
                'sparsity': -1, 'error': 'gurobipy not installed'}

    c = prob['c']; A = prob['A_csr']; b = prob['b']; n = prob['n']

    t0 = time.time()
    model = gp.Model("exp")
    model.setParam('OutputFlag', 0)
    model.setParam('Method', method)
    model.setParam('Threads', threads)
    model.setParam('TimeLimit', 300)

    x = model.addMVar(n, lb=0.0, ub=1.0)
    model.addConstr(x.sum() == 1)
    model.addMConstr(A, x, '<', b)
    model.setObjective(c @ x, gp.GRB.MINIMIZE)
    model.optimize()

    elapsed = time.time() - t0
    if model.status == gp.GRB.OPTIMAL:
        xv = x.X
        return {
            'solver': 'Gurobi',
            'objective': float(model.ObjVal),
            'time': elapsed,
            'sparsity': int(np.sum(xv > 1e-10)),
            'iter_count': int(model.IterCount),
        }
    else:
        return {'solver': 'Gurobi', 'objective': np.inf, 'time': elapsed,
                'sparsity': -1, 'status': model.status}


def solve_highs(prob: Dict) -> Dict:
    """scipy HiGHS (免费基线)"""
    c = prob['c']; A = prob['A_csr']; b = prob['b']; n = prob['n']

    t0 = time.time()
    if n >= 100000:
        A_eq = sparse.csr_matrix(([1.0]*n, ([0]*n, range(n))), shape=(1, n))
        res = linprog(c, A_ub=A, b_ub=b,
                      A_eq=A_eq, b_eq=np.array([1.0]),
                      bounds=(0, None), method='highs')
    else:
        res = linprog(c, A_ub=A if sparse.issparse(A) else A, b_ub=b,
                      A_eq=np.ones((1, n)), b_eq=np.array([1.0]),
                      bounds=(0, None), method='highs')

    elapsed = time.time() - t0
    if res.success:
        return {
            'solver': 'HiGHS',
            'objective': float(res.fun),
            'time': elapsed,
            'sparsity': int(np.sum(res.x > 1e-10)),
            'nit': res.nit,
        }
    else:
        return {'solver': 'HiGHS', 'objective': np.inf, 'time': elapsed,
                'sparsity': -1, 'status': res.message}


def solve_ga01(prob: Dict) -> Dict:
    """0-1 ImprovedGA (消融基线)"""
    from final_benchmark import Improved01GA
    n = prob['n']
    if n <= 1000:
        pop, gen = 120, 120
    elif n <= 10000:
        pop, gen = 150, 180
    elif n <= 100000:
        pop, gen = 180, 200
    else:
        pop, gen = 200, 250

    ga = Improved01GA(prob, pop_size=pop, max_gen=gen, K=prob['m'] + 3)
    result = ga.run()
    return {
        'solver': '01GA',
        'objective': result['objective'],
        'time': result['time'],
        'sparsity': result['sparsity'],
        'lp_calls': result.get('lp_calls', 0),
    }


# ============================================================================
# 实验主流程
# ============================================================================

SOLVERS = {
    'S-CG': solve_scg,
    'Gurobi': lambda p: solve_gurobi(p, method=-1, threads=1),
    'HiGHS': solve_highs,
    '01GA': solve_ga01,
}


def run_single_experiment(config: Dict, solvers: List[str] = None) -> Dict:
    """运行单次实验"""
    if solvers is None:
        solvers = ['S-CG', 'Gurobi', 'HiGHS']

    n = config['n']; m = config['m']
    seed = config['seed']; tightness = config['tightness']

    prob = generate_problem(n=n, m=m, seed=seed, slack_tightness=tightness)

    # 诊断问题难度
    from problem_factory import diagnose_problem
    diag = diagnose_problem(prob)

    result = {
        'n': n, 'm': m, 'tightness': tightness,
        'trial': config['trial'], 'seed': seed,
        'difficulty': diag['difficulty'],
        'n_single_feasible': diag['n_single_feasible'],
        'lp_nnz': diag['lp_nnz'],
        'reference_obj': diag['obj_lp'],  # 真值 (HiGHS full LP)
    }

    for sname in solvers:
        if sname in SOLVERS:
            try:
                sresult = SOLVERS[sname](prob)
                result[sname.lower() + '_obj'] = sresult['objective']
                result[sname.lower() + '_time'] = sresult['time']
                result[sname.lower() + '_nnz'] = sresult['sparsity']
                if 'method' in sresult:
                    result[sname.lower() + '_method'] = sresult['method']
                if 'iterations' in sresult:
                    result[sname.lower() + '_iters'] = sresult['iterations']
                if 'nit' in sresult:
                    result[sname.lower() + '_nit'] = sresult['nit']
            except Exception as e:
                print(f"  ERROR [{sname}]: {e}")
                result[sname.lower() + '_obj'] = np.inf
                result[sname.lower() + '_time'] = np.inf

    # 计算 gap (相对于 HiGHS 完整LP真值)
    ref = result['reference_obj']
    for sname in solvers:
        obj_key = sname.lower() + '_obj'
        if obj_key in result and not np.isinf(ref) and not np.isinf(result[obj_key]):
            result[sname.lower() + '_gap'] = (
                abs(result[obj_key] - ref) / max(abs(ref), 1e-15) * 100)

    return result


def run_grid(n_list, m_list, tightness_list, n_trials=10,
             solvers=None, output_csv=None):
    """运行参数网格实验"""
    configs = problem_grid(n_list, m_list, tightness_list, n_trials)
    results = []

    print(f"Running {len(configs)} experiments...")
    print(f"  n: {n_list}")
    print(f"  m: {m_list}")
    print(f"  tightness: {tightness_list}")
    print(f"  trials per config: {n_trials}")
    print(f"  solvers: {solvers or list(SOLVERS.keys())}")

    for i, cfg in enumerate(configs):
        r = run_single_experiment(cfg, solvers)
        results.append(r)

        if (i + 1) % max(1, len(configs) // 20) == 0:
            print(f"  [{i+1}/{len(configs)}] n={cfg['n']} m={cfg['m']} "
                  f"trial={cfg['trial']} diff={r['difficulty']} "
                  f"S-CG={r.get('s-cg_obj', np.inf):.6f}")

    print(f"Done. {len(results)} experiments completed.")

    if output_csv:
        save_results(results, output_csv)

    return results


def save_results(results: List[Dict], filepath: str):
    """保存结果到CSV"""
    if not results:
        return
    keys = results[0].keys()
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {filepath}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='S-CG Experiment Runner')
    parser.add_argument('--quick', action='store_true',
                        help='Quick run: 1 trial, limited grid')
    parser.add_argument('--full', action='store_true',
                        help='Full run: 10 trials, complete grid')
    parser.add_argument('--n', type=int, nargs='+',
                        default=[100, 1000, 10000, 100000, 1000000],
                        help='Variable dimensions')
    parser.add_argument('--m', type=int, nargs='+', default=[5, 10, 20, 50],
                        help='Constraint counts')
    parser.add_argument('--tightness', type=float, nargs='+',
                        default=[0.01, 0.05, 0.1],
                        help='Slack tightness levels')
    parser.add_argument('--trials', type=int, default=None,
                        help='Number of trials per config')
    parser.add_argument('--output', type=str, default='results/experiment_a.csv',
                        help='Output CSV path')
    parser.add_argument('--solvers', type=str, nargs='+',
                        default=['S-CG', 'Gurobi', 'HiGHS'],
                        help='Solvers to compare')
    parser.add_argument('--no-gurobi', action='store_true',
                        help='Skip Gurobi (if license unavailable)')

    args = parser.parse_args()

    if args.quick:
        n_trials = 1
        n_list = [100, 1000, 10000, 100000]
        m_list = [10]
        tightness_list = [0.1]
    elif args.full:
        n_trials = args.trials or 10
        n_list = args.n
        m_list = args.m
        tightness_list = args.tightness
    else:
        n_trials = args.trials or 3
        n_list = args.n
        m_list = args.m
        tightness_list = args.tightness

    solvers = args.solvers
    if args.no_gurobi and 'Gurobi' in solvers:
        solvers.remove('Gurobi')

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.',
                exist_ok=True)

    results = run_grid(n_list, m_list, tightness_list, n_trials,
                       solvers=solvers, output_csv=args.output)

    # Quick summary
    from experiments.stats_utils import summarize
    summarize(results)

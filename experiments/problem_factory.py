"""
================================================================================
受控问题生成器 — 稀疏单纯形LP

支持:
  - 控制 slack_tightness (退化程度)
  - 控制 A 密度
  - 自动分类问题难度 (degenerate / medium / hard)
  - 已知真解 (通过完整LP验证)
================================================================================
"""
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
import time
from typing import Dict, Tuple, Optional


def generate_problem(n: int, m: int = 10, seed: int = 42,
                     density: Optional[float] = None,
                     slack_tightness: float = 0.1) -> Dict:
    """
    生成稀疏单纯形LP问题实例

    Args:
        n: 变量数
        m: 约束数
        seed: 随机种子
        density: A 的密度 (None = 自适应)
        slack_tightness: 约束紧度
            0.001-0.01: 极紧 → 几乎无单列可行 → HARD
            0.05:       中等 → 部分单列可行 → MEDIUM
            0.1-0.5:    宽松 → 大量单列可行 → EASY (degenerate)

    Returns:
        {'c': [n], 'A_csr': sparse, 'A_csc': sparse, 'b': [m],
         'n': n, 'm': m, 'x_true': [n], 'true_nnz': int,
         'slack_tightness': float, 'density': float}
    """
    rng = np.random.RandomState(seed)

    # 真解: 稀疏 (m+1 个非零分量)
    nnz_true = min(m + 1, n)
    nz_idx = rng.choice(n, nnz_true, replace=False)
    x_true = np.zeros(n)
    x_true[nz_idx] = np.abs(rng.randn(nnz_true))
    x_true /= x_true.sum()

    # 稀疏约束矩阵
    if density is None:
        if n <= 1000:
            density = 0.5
        elif n <= 10000:
            density = 0.05
        else:
            density = 0.005

    A = sparse.random(m, n, density=density, format='csr',
                      data_rvs=lambda sz: np.abs(rng.randn(sz)) * 0.5,
                      random_state=rng)

    # 约束右端: b = A@x_true + slack
    slack = np.abs(rng.rand(m)) * slack_tightness + slack_tightness / 2
    b = A @ x_true + slack

    # 目标函数: 使真解有利
    c = rng.randn(n) * 2.0
    c[nz_idx] = np.abs(c[nz_idx])

    return {
        'c': c, 'A_csr': A, 'A_csc': A.tocsc(), 'b': b,
        'n': n, 'm': m,
        'x_true': x_true, 'true_nnz': nnz_true,
        'slack_tightness': slack_tightness,
        'density': density,
    }


def diagnose_problem(prob: Dict) -> Dict:
    """
    诊断问题难度

    Returns:
        {'n_single_feasible': int,             # 可行单列数
         'best_single_obj': float,              # 最优单列目标值
         'obj_true': float,                     # 真解目标值
         'difficulty': str,                     # 'degenerate' / 'medium' / 'hard'
         'is_degenerate': bool,                 # 是否退化为单列
         'lp_obj': float, 'lp_nnz': int}        # 完整LP验证
    """
    c = prob['c']; A = prob['A_csr']; b = prob['b']
    n = prob['n']; m = prob['m']

    # 单列可行性扫描
    feasible = np.ones(n, dtype=bool)
    A_csr = A.tocsr()
    for i in range(m):
        row = A_csr.getrow(i)
        for j, val in zip(row.indices, row.data):
            if val > b[i]:
                feasible[j] = False

    n_feas = feasible.sum()
    best_obj = float(c[feasible].min()) if n_feas > 0 else np.inf

    # 完整LP验证
    t0 = time.time()
    if n >= 100000:
        A_eq = sparse.csr_matrix(([1.0]*n, ([0]*n, range(n))), shape=(1, n))
        res = linprog(c, A_ub=A, b_ub=b,
                      A_eq=A_eq, b_eq=np.array([1.0]),
                      bounds=(0, None), method='highs')
    else:
        res = linprog(c, A_ub=A.toarray() if sparse.issparse(A) else A, b_ub=b,
                      A_eq=np.ones((1, n)), b_eq=np.array([1.0]),
                      bounds=(0, None), method='highs')

    lp_time = time.time() - t0
    lp_obj = res.fun if res.success else np.inf
    lp_nnz = int(np.sum(res.x > 1e-10)) if res.success else -1

    # 分类
    if n_feas > 0 and abs(best_obj - lp_obj) < 1e-8:
        difficulty = 'degenerate'
        is_degenerate = True
    elif n_feas == 0:
        difficulty = 'hard'
        is_degenerate = False
    else:
        difficulty = 'medium'
        is_degenerate = False

    return {
        'n_single_feasible': int(n_feas),
        'best_single_obj': best_obj,
        'obj_true': float(c @ prob['x_true']),
        'obj_lp': lp_obj,
        'lp_nnz': lp_nnz,
        'lp_time': lp_time,
        'difficulty': difficulty,
        'is_degenerate': is_degenerate,
    }


def problem_grid(n_list, m_list, tightness_list, seeds_per_config=10):
    """生成实验参数网格"""
    configs = []
    for n in n_list:
        for m in m_list:
            for tightness in tightness_list:
                for trial in range(seeds_per_config):
                    seed = 1000 * int(np.log10(max(n, 10))) + 100 * m + trial
                    configs.append({
                        'n': n, 'm': m, 'tightness': tightness,
                        'trial': trial, 'seed': seed,
                    })
    return configs

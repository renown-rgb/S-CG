"""
================================================================================
通用稀疏可行/优化问题求解器

问题:
  min  cᵀx  (或仅 feasibility)
  s.t. Ax ≤ b,  ‖x‖₀ ≤ K
  x ∈ ℝⁿ  (无符号/范围限制)

核心: 支撑集 S 上的子问题 = 小型 LP，|S| ≤ K ≪ n

策略:
  Level 0: 若 K ≥ m+1 → LP 放宽有稀疏解 (类似之前，但有符号)
  Level 1: 列扫描——单列/双列可行性
  Level 2: 候选池枚举——贪心构建候选集 + K-子集枚举
  Level 3: 0-1 GA — 大候选池搜索
================================================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import linprog
from scipy import sparse
import time
from itertools import combinations
from typing import Dict, Set, Optional


class GeneralSparseSolver:
    """
    通用 ‖x‖₀ ≤ K 稀疏求解器 (x 无符号限制)

    Parameters
    ----------
    K : int
        非零分量数上限
    objective : str
        'feasibility': 仅求可行解
        'minimize':   min cᵀx (需提供 c)
    max_candidates : int
        候选池大小上限
    """

    def __init__(self, K: int, objective='feasibility',
                 max_candidates=200, verbose=False):
        if objective not in ('feasibility', 'minimize'):
            raise ValueError("objective must be 'feasibility' or 'minimize'")
        self.K = K
        self.objective = objective
        self.max_candidates = max_candidates
        self.verbose = verbose

    # ==================================================================
    # 主入口
    # ==================================================================

    def solve(self, A, b, c=None):
        """
        求解稀疏可行/优化问题

        Parameters
        ----------
        A : (m, n) sparse/dense matrix
        b : (m,) array
        c : (n,) array or None (仅 minimize 模式需要)

        Returns
        -------
        dict with keys: x, objective, feasible, method, time, ...
        """
        t0 = time.time()
        n = A.shape[1]
        m = len(b)
        has_obj = self.objective == 'minimize' and c is not None

        if self.verbose:
            mode = 'min cᵀx' if has_obj else 'feasibility'
            print(f"[GeneralSparse] n={n}, m={m}, K={self.K}, {mode}")

        # ---- 核心洞察: LP 基本可行解天然 ‖x‖₀ ≤ m ----
        # 转换为标准形: min cᵀ(x⁺-x⁻) s.t. A(x⁺-x⁻)+Is=b, x⁺,x⁻,s≥0
        # 共 2n+m 变量, m 等式约束. BFS 至多 m 个基本变量.
        # 每个 x_j≠0 消耗至少 1 个基本 → ‖x‖₀ ≤ m
        # 因此 K ≥ m 时问题退化为 P
        if self.K >= m:
            result = self._solve_unrestricted_lp(A, b, c, has_obj)
            result['method'] = 'lp_natural_sparsity'
            result['sparsity_bound'] = m
            return result

        # ---- K < m: NP-hard regime ----
        # 核心策略: LP 解 x* 的支撑 S* 满足 |S*| ≤ m.
        # 最优 K-稀疏解的支撑要么在 S* 内 (组合数 C(|S*|,K) ≤ C(m,K)),
        # 要么在 S* 的邻域内.
        # 因此从 S* 出发, 而非从 n 个变量中盲目搜索.

        # 先解 LP 得 x* 和支撑 S*
        lp_result = self._solve_unrestricted_lp(A, b, c, has_obj)
        if lp_result['feasible']:
            S_star = set(np.where(np.abs(lp_result['x']) > 1e-10)[0])
            s_star = len(S_star)
            if self.verbose:
                print(f"  LP: f*={lp_result['objective']:.4f}, |S*|={s_star}")

            if s_star <= self.K:
                result = lp_result
                result['method'] = 'lp_already_k_sparse'
                result['time'] = time.time() - t0
                return result

            # 从 S* 的 K-子集入手
            # 从 S* 枚举 K-子集: C(s_star, K)
            n_combos_s = _n_choose_k(s_star, self.K)
            if n_combos_s <= 100000:  # ≤10万组合, 直接枚举
                # S* 足够小: 直接枚举 K-子集
                if self.verbose:
                    print(f"  Enumerating C({s_star},{self.K}) subsets from LP support")
                result = self._enumerate(A, b, c, S_star, has_obj)
                if result['feasible']:
                    result['method'] = 'enumerate_lp_support'
                    result['time'] = time.time() - t0
                    return result

        # 扩展候选池: S* + 其近邻
        candidates = set(S_star) if 'S_star' in dir() else set()
        extra = self._build_candidate_pool(A, b, c, n, m, has_obj)
        candidates.update(extra)
        candidates = set(sorted(candidates)[:self.max_candidates])

        pool_size = len(candidates)
        if self.verbose:
            print(f"  Pool: {pool_size} (LP support + candidates)")

        n_combos = _n_choose_k(pool_size, min(self.K, pool_size))
        if n_combos <= 3000:
            result = self._enumerate(A, b, c, candidates, has_obj)
        else:
            result = self._ga_search(A, b, c, candidates, has_obj)

        result['time'] = time.time() - t0
        return result

    # ==================================================================
    # Level 1: 单列/双列扫描
    # ==================================================================

    def _single_column_scan(self, A, b, c, has_obj):
        """扫描可行单列 (含符号)"""
        n = A.shape[1]

        if sparse.issparse(A):
            A_csr = A.tocsr()
        else:
            A_csr = A

        # 正方向: x_j ≥ 0 → A_{:,j} * x_j ≤ b
        # 负方向: x_j ≤ 0 → -A_{:,j} * (-x_j) ≤ b, 即 A_{:,j} * x_j ≤ b
        # 关键: 若 A_{:,j} 的所有元素同号 (或零), 则存在标量 t 使 t*A_{:,j} ≤ b

        for j in range(n):
            col = A_csr[:, j].toarray().flatten() if sparse.issparse(A_csr) else A_csr[:, j]

            # 尝试向正方向缩放
            x_j_pos = self._max_feasible_scale(col, b)
            if np.isfinite(x_j_pos) and abs(x_j_pos) > 1e-15:
                x = np.zeros(n)
                x[j] = x_j_pos
                obj = float(c @ x) if has_obj else 0.0
                return {'x': x, 'objective': obj, 'feasible': True,
                        'sparsity': 1, 'method': 'single_column'}

            # 尝试向负方向缩放
            x_j_neg = self._max_feasible_scale(-col, b)
            if np.isfinite(x_j_neg) and abs(x_j_neg) > 1e-15:
                x = np.zeros(n)
                x[j] = -x_j_neg
                obj = float(c @ x) if has_obj else 0.0
                return {'x': x, 'objective': obj, 'feasible': True,
                        'sparsity': 1, 'method': 'single_column'}

        return None

    def _pair_scan(self, A, b, c, has_obj):
        """扫描双列组合 (用于 K ≥ 2)"""
        n = A.shape[1]
        if sparse.issparse(A):
            A_d = A.toarray()
        else:
            A_d = A

        # 仅在成本最低的 100 列上搜索
        if has_obj and c is not None:
            cheap_idx = np.argsort(np.abs(c))[:100]
        else:
            cheap_idx = np.random.choice(n, min(100, n), replace=False)

        for i_idx, i in enumerate(cheap_idx):
            for j in cheap_idx[i_idx + 1:]:
                # 求解双变量 LP: min/max over x_i, x_j
                col_i = A_d[:, i]
                col_j = A_d[:, j]

                # 简单检查: 线性组合 col_i * t_i + col_j * t_j ≤ b 是否有解
                # 等价于 LP: min 0 s.t. col_i * x_i + col_j * x_j ≤ b
                res = linprog(np.zeros(2),
                              A_ub=np.column_stack([col_i, col_j]),
                              b_ub=b,
                              bounds=[(None, None), (None, None)],
                              method='highs')
                if res.success:
                    x = np.zeros(n)
                    x[i] = res.x[0]
                    x[j] = res.x[1]
                    obj = float(c @ x) if has_obj and c is not None else 0.0
                    nnz = int(np.sum(np.abs(x) > 1e-10))
                    if nnz <= self.K:
                        return {'x': x, 'objective': obj, 'feasible': True,
                                'sparsity': nnz, 'method': 'pair_scan'}

        return None

    def _max_feasible_scale(self, col, b):
        """找最大可行正标量 t 使 t*col ≤ b"""
        # t*col[i] ≤ b[i] for all i
        # 若 col[i] > 0: t ≤ b[i]/col[i]
        # 若 col[i] < 0: t ≥ b[i]/col[i] (t≥0时自动满足, t 无上界)
        # 若 col[i] = 0: 0 ≤ b[i] 恒成立 (若 b[i] < 0 则不可行)

        t_max = np.inf
        for i in range(len(col)):
            if col[i] > 1e-15:
                t_max = min(t_max, b[i] / col[i])
            elif col[i] < -1e-15:
                # col[i] < 0, t * col[i] ≤ b[i]
                # 因为 col[i] < 0, t ≥ 0 → t*col[i] ≤ 0
                # 若 b[i] ≥ 0, 恒成立; 若 b[i] < 0, 需 t ≥ b[i]/col[i]
                # 但 col[i] < 0, 所以 b[i]/col[i] > 0 (若 b[i] < 0)
                pass  # t≥0 constraint will handle
            else:
                if b[i] < -1e-15:
                    return -np.inf  # 不可行

        if np.isfinite(t_max) and t_max > 1e-15:
            return t_max * 0.5  # 取一半留余地
        return -np.inf

    # ==================================================================
    # 候选池构建
    # ==================================================================

    def _build_candidate_pool(self, A, b, c, n, m, has_obj):
        """构建候选变量池"""
        candidates = set()
        A_d = A.toarray() if sparse.issparse(A) else A

        # 策略1: 每列单独评估"可行性距离"
        scores = []
        for j in range(n):
            col = A_d[:, j]
            # 到可行区域的最小距离 (简单启发式)
            violations_pos = np.maximum(0, col - b).sum()
            violations_neg = np.maximum(0, -col - b).sum()
            score = min(violations_pos, violations_neg)
            scores.append((score, j))

        scores.sort()
        for _, j in scores[:self.max_candidates // 2]:
            candidates.add(j)

        # 策略2: 若有目标函数, 加入成本最低的变量
        if has_obj and c is not None:
            cheap = np.argsort(c)[:self.max_candidates // 2]
            candidates.update(int(j) for j in cheap)

        # 策略3: 加入 A 列范数最大的变量 (更有"影响力")
        if sparse.issparse(A):
            col_norms = np.array([A[:, j].power(2).sum() for j in range(n)]).flatten()
        else:
            col_norms = np.sum(A ** 2, axis=0)
        top_norm = np.argsort(-col_norms)[:self.max_candidates // 4]
        candidates.update(int(j) for j in top_norm)

        return candidates

    # ==================================================================
    # 子集搜索
    # ==================================================================

    def _enumerate(self, A, b, c, candidates, has_obj):
        """枚举 K-子集"""
        cand_list = sorted(candidates)
        K = min(self.K, len(cand_list))
        best_obj = np.inf
        best_x = None
        n_eval = 0

        if self.verbose:
            print(f"    Enumerating C({len(cand_list)},{K}) subsets...")

        for subset in combinations(cand_list, K):
            n_eval += 1
            x, obj, feas = self._feasibility_on_support(A, b, c, set(subset), has_obj)
            if feas:
                if not has_obj or obj < best_obj - 1e-12:
                    best_obj = obj
                    best_x = x
                    if not has_obj:
                        break  # feasibility only: 找到就停

        return {
            'x': best_x if best_x is not None else np.zeros(A.shape[1]),
            'objective': best_obj,
            'feasible': best_x is not None,
            'sparsity': int(np.sum(np.abs(best_x) > 1e-10)) if best_x is not None else 0,
            'method': 'enumerate',
            'subsets_evaluated': n_eval,
        }

    def _ga_search(self, A, b, c, candidates, has_obj):
        """GA 在候选池搜索"""
        n = A.shape[1]
        cand_arr = np.array(sorted(candidates))
        pool_size = len(cand_arr)
        K = min(self.K, pool_size)
        A_d = A.toarray() if sparse.issparse(A) else A

        pop_size = min(60, max(20, pool_size // 2))
        max_gen = min(80, pool_size)
        rng = np.random.RandomState(123)

        # 初始化
        pop = []
        for _ in range(pop_size):
            chosen = rng.choice(pool_size, K, replace=False)
            pop.append(set(int(cand_arr[i]) for i in chosen))

        # 评估
        fit = {}; feas = {}
        for s in pop:
            key = tuple(sorted(s))
            x, obj, f = self._feasibility_on_support(A_d, b, c, s, has_obj)
            fit[key] = obj; feas[key] = f

        best_x = None; best_obj = np.inf
        for s in pop:
            key = tuple(sorted(s))
            if feas[key]:
                if not has_obj or fit[key] < best_obj - 1e-12:
                    best_obj = fit[key]; best_x = self._rebuild_x(A.shape[1], s, A_d, b)

        no_imp = 0
        for gen in range(max_gen):
            ranked = sorted(pop, key=lambda s: (not feas.get(tuple(sorted(s)), False),
                                                 fit.get(tuple(sorted(s)), np.inf)))
            elites = [set(s) for s in ranked[:max(3, pop_size // 10)]]
            offspring = []

            while len(offspring) < pop_size - len(elites):
                p1 = rng.choice(pop_size, 1)[0]
                p2 = rng.choice(pop_size, 1)[0]
                union = list(pop[p1] | pop[p2])
                if len(union) >= K:
                    child = set(int(v) for v in rng.choice(union, K, replace=False))
                else:
                    child = set(int(v) for v in rng.choice(cand_arr, K, replace=False))
                # 变异
                if rng.random() < 0.5:
                    n_mut = max(1, rng.poisson(1.0))
                    clist = list(child)
                    if len(clist) >= n_mut:
                        child -= set(rng.choice(clist, n_mut, replace=False))
                        child.update(int(cand_arr[i]) for i in rng.choice(pool_size, n_mut, replace=False))
                offspring.append(child)

            pop = elites + offspring
            for s in pop:
                key = tuple(sorted(s))
                if key not in fit:
                    x, obj, f = self._feasibility_on_support(A_d, b, c, s, has_obj)
                    fit[key] = obj; feas[key] = f

            for s in pop:
                key = tuple(sorted(s))
                if feas[key] and (not has_obj or fit[key] < best_obj - 1e-12):
                    best_obj = fit[key]
                    best_x = self._rebuild_x(A.shape[1], s, A_d, b)
                    no_imp = 0
            no_imp += 1
            if no_imp > 25:
                break

        feasible = best_x is not None
        return {
            'x': best_x if feasible else np.zeros(n),
            'objective': best_obj,
            'feasible': feasible,
            'sparsity': int(np.sum(np.abs(best_x) > 1e-10)) if feasible else 0,
            'method': 'ga',
            'generations': gen + 1,
            'lp_calls': len(fit),
        }

    # ==================================================================
    # 辅助
    # ==================================================================

    def _feasibility_on_support(self, A, b, c, support, has_obj):
        """在支撑集上求解 LP"""
        arr = np.array(list(support))
        if len(arr) == 0:
            return None, np.inf, False

        try:
            A_sub = A[:, arr]
            if sparse.issparse(A_sub):
                A_sub = A_sub.toarray()

            if has_obj and c is not None:
                # min c[arr]ᵀx_sub
                res = linprog(c[arr], A_ub=A_sub, b_ub=b,
                              bounds=[(None, None)] * len(arr), method='highs')
            else:
                # feasibility: min 0
                res = linprog(np.zeros(len(arr)), A_ub=A_sub, b_ub=b,
                              bounds=[(None, None)] * len(arr), method='highs')

            if res.success:
                x_full = np.zeros(A.shape[1])
                x_full[arr] = res.x
                obj = float(c @ x_full) if has_obj and c is not None else 0.0
                return x_full, obj, True
        except:
            pass

        return None, np.inf, False

    def _rebuild_x(self, n, support, A, b):
        """从支撑集重建完整解"""
        arr = np.array(list(support))
        res = linprog(np.zeros(len(arr)), A_ub=A[:, arr],
                      b_ub=b, bounds=[(None, None)] * len(arr), method='highs')
        if res.success:
            x = np.zeros(n)
            x[arr] = res.x
            return x
        return None

    def _solve_unrestricted_lp(self, A, b, c, has_obj):
        """K ≥ m+1: 直接解 LP (无 ‖x‖₀ 约束)"""
        n = A.shape[1]
        if has_obj and c is not None:
            res = linprog(c, A_ub=A, b_ub=b,
                          bounds=[(None, None)] * n, method='highs')
        else:
            res = linprog(np.zeros(n), A_ub=A, b_ub=b,
                          bounds=[(None, None)] * n, method='highs')

        if res.success:
            x = res.x
            x[np.abs(x) < 1e-14] = 0
            return {
                'x': x, 'objective': float(c @ x) if has_obj and c is not None else 0.0,
                'feasible': True,
                'sparsity': int(np.sum(np.abs(x) > 1e-10)),
                'method': 'lp_natural_sparsity',
                'time': 0.0,
            }
        return {'x': None, 'objective': np.inf, 'feasible': False,
                'sparsity': -1, 'method': 'lp_natural_sparsity'}


def _n_choose_k(n, k):
    if k > n:
        return float('inf')
    k = min(k, n - k)
    result = 1.0
    for i in range(k):
        result = result * (n - i) / (i + 1)
        if result > 1e9:
            return int(1e9)
    return int(result)


# ============================================================================
# 实验
# ============================================================================

def benchmark():
    """对比: 通用稀疏求解器 vs Gurobi MIP"""
    from final_benchmark import generate_problem

    print("=" * 70)
    print("  General Sparse Feasibility Solver")
    print("=" * 70)

    configs = [
        # (n, m, K)
        (100, 20, 4),
        (200, 20, 5),
        (500, 30, 4),
        (1000, 30, 5),
    ]

    for n, m, K in configs:
        print(f"\n{'='*50}")
        print(f"n={n}, m={m}, K={K}")
        print(f"{'='*50}")

        for trial in range(3):
            seed = 20000 + trial * 100
            prob = generate_problem(n=n, m=m, seed=seed)
            A = prob['A_csr']
            b = prob['b']
            c = prob['c']

            # 我们的求解器 (feasibility mode)
            solver = GeneralSparseSolver(K=K, objective='feasibility',
                                          max_candidates=150, verbose=False)
            t0 = time.time()
            result = solver.solve(A, b)
            t_ours = time.time() - t0

            # Gurobi MIP baseline
            try:
                import gurobipy as gp
                t0 = time.time()
                model = gp.Model("g")
                model.setParam('OutputFlag', 0)
                model.setParam('Threads', 1)
                model.setParam('TimeLimit', 60)

                xv = model.addMVar(n, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY,
                                    vtype=gp.GRB.CONTINUOUS)
                # Split variable: x⁺, x⁻ ≥ 0, x = x⁺ - x⁻
                xp = model.addMVar(n, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="xp")
                xm = model.addMVar(n, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="xm")
                model.addConstr(xv == xp - xm)

                # Cardinality: z_i = 1 if |x_i| > 0
                z = model.addMVar(n, vtype=gp.GRB.BINARY)
                model.addConstr(xp + xm <= z)  # if z_i=0 then xp_i = xm_i = 0
                model.addConstr(z.sum() <= K)

                model.addMConstr(A, xv, '<', b)
                model.optimize()

                t_mip = time.time() - t0
                feas_mip = model.status == gp.GRB.OPTIMAL
                print(f"  trial {trial}: ours={'FOUND' if result['feasible'] else 'NONE'} "
                      f"({t_ours*1000:.1f}ms) "
                      f"Gurobi={'FOUND' if feas_mip else 'NOT FOUND'} "
                      f"({t_mip:.1f}s)")
            except ImportError:
                print(f"  trial {trial}: ours={'FOUND' if result['feasible'] else 'NONE'} "
                      f"({t_ours*1000:.1f}ms) [Gurobi N/A]")


if __name__ == "__main__":
    benchmark()

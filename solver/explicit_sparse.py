"""
================================================================================
显式稀疏约束 LP 求解器 — ‖x‖₀ ≤ K

问题:
  min cᵀx  s.t. Ax ≤ b, Σx=1, x≥0, ‖x‖₀ ≤ K

策略: 两级——免费LP下界 + K-子集搜索

核心洞察:
  1. 若 K ≥ m+1: 免费稀疏性适用, S-CG 直接解 (P)
  2. 若 K < m+1: NP-hard. 但 m+1 很小 (≤ 50), 枚举空间可控.
     先解免费LP得下界 f* 和支撑 S* (|S*|≤m+1).
     最优 K-稀疏解的支撑必为 S* 的某个 K-子集, 或
     来自扩展候选池的 K-子集.

算法:
  Phase 1: 求解免费 LP (S-CG) → 下界 f*, 支撑 S*
  Phase 2: 构建候选池 (S* + S-CG Level 1 可行单列 + 低成本列)
  Phase 3: 在候选池的 K-子集上搜索 (枚举 / GA / 贪心)
================================================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import linprog
from scipy import sparse
import time
from itertools import combinations
from typing import Dict, List, Tuple, Set, Optional

from fast_solver import FastSolver, SingleColumnSolver


class ExplicitSparseSolver:
    """
    显式 ‖x‖₀ ≤ K 约束的稀疏 LP 求解器

    Parameters
    ----------
    K : int
        非零分量数上限
    method : str
        'enumerate': 枚举候选K-子集 (小候选池推荐)
        'ga': 0-1 遗传算法 (大候选池)
        'greedy': 贪心构造 (快速近似)
        'auto': 自动选择 (默认)
    max_candidates : int
        候选池大小上限
    """

    def __init__(self, K: int, method='auto', max_candidates=200,
                 verbose=False):
        self.K = K
        self.method = method
        self.max_candidates = max_candidates
        self.verbose = verbose
        self.free_solver = FastSolver(verbose=False)

    def solve(self, c, A, b):
        """
        求解显式 ‖x‖₀ ≤ K 约束的稀疏 LP
        """
        t0 = time.time()
        n, m = len(c), len(b)

        # ---- 若 K ≥ m+1: 免费稀疏性 ----
        if self.K >= m + 1:
            if self.verbose:
                print(f"[Explicit] K={self.K} ≥ m+1={m+1}, using free sparsity")
            result = self.free_solver.solve(c, A, b)
            result['method'] = 'free_sparsity'
            result['support_size'] = result['sparsity']
            return result

        # ---- Phase 1: 解免费 LP, 得下界 ----
        if self.verbose:
            print(f"[Explicit] K={self.K} < m+1={m+1}, NP-hard regime")
            print(f"  Phase 1: solving free LP...")

        free_result = self.free_solver.solve(c, A, b)
        f_lower = free_result['objective']
        S_star = set(np.where(free_result['x'] > 1e-10)[0])
        s_star = len(S_star)

        if self.verbose:
            print(f"  Free LP: f*={f_lower:.6f}, support={s_star}")

        # 若免费LP的解已满足K-稀疏
        if s_star <= self.K:
            if self.verbose:
                print(f"  Free LP already K-sparse!")
            free_result['method'] = 'free_sparsity_already_sparse'
            return free_result

        # ---- Phase 2: 构建候选池 ----
        if self.verbose:
            print(f"  Phase 2: building candidate pool...")

        candidates = self._build_candidate_pool(c, A, b, S_star, n, m)

        if self.verbose:
            print(f"  Candidate pool: {len(candidates)} variables")

        # ---- Phase 3: 搜索最优 K-子集 ----
        if self.verbose:
            print(f"  Phase 3: searching optimal K-subset...")

        best_result = self._search_k_subset(c, A, b, candidates, f_lower)

        elapsed = time.time() - t0
        best_result['time'] = elapsed
        best_result['free_lower_bound'] = f_lower
        best_result['free_support_size'] = s_star
        best_result['candidate_pool_size'] = len(candidates)

        # Gap from lower bound
        f_upper = best_result['objective']
        gap = abs(f_upper - f_lower) / max(abs(f_lower), 1e-15) * 100
        best_result['optimality_gap_pct'] = gap

        if self.verbose:
            print(f"  Result: f={f_upper:.6f}, nnz={best_result['sparsity']}, "
                  f"gap_from_lower={gap:.2f}%, time={elapsed:.3f}s")

        return best_result

    def _build_candidate_pool(self, c, A, b, S_star, n, m):
        """构建候选变量池"""

        candidates = set(S_star)  # 免费LP支撑必含

        # 添加 Level 1 的可行单列 (按成本排序取前若干)
        x_sc, obj_sc, _, _, _ = SingleColumnSolver.solve(
            c, A, b, verify_with_lp=False)
        if not np.isinf(obj_sc):
            # 成本最低的若干可行单列
            n_feas_add = min(50, n)
            cheap_feasible = []
            for j in range(n):
                if x_sc[j] > 0.5 or (not np.isinf(obj_sc) and c[j] < obj_sc + 0.1):
                    cheap_feasible.append((c[j], j))

            cheap_feasible.sort()
            for _, j in cheap_feasible[:n_feas_add]:
                if len(candidates) < self.max_candidates:
                    candidates.add(j)

        # 添加成本最低的变量
        n_cheap = min(100, n)
        cheap_idx = np.argsort(c)[:n_cheap]
        for j in cheap_idx:
            if len(candidates) < self.max_candidates:
                candidates.add(j)

        return candidates

    def _search_k_subset(self, c, A, b, candidates, f_lower):
        """搜索最优 K-子集"""

        cand_list = sorted(candidates)
        n_cand = len(cand_list)

        # 自适应选择方法
        method = self.method
        if method == 'auto':
            # 若 C(n_cand, K) ≤ 5000, 枚举; 否则 GA
            n_combos = self._n_choose_k(n_cand, min(self.K, n_cand))
            method = 'enumerate' if n_combos <= 5000 else 'ga'

        if self.verbose:
            print(f"    Method: {method} ({n_cand} choose {self.K})")

        if method == 'enumerate':
            return self._enumerate_k_subsets(c, A, b, cand_list, f_lower)
        elif method == 'ga':
            return self._ga_search(c, A, b, cand_list, f_lower)
        else:  # greedy
            return self._greedy_search(c, A, b, cand_list)

    def _enumerate_k_subsets(self, c, A, b, cand_list, f_lower):
        """枚举所有 K-子集"""
        K = min(self.K, len(cand_list))
        best_obj = np.inf
        best_x = None

        n_combos = self._n_choose_k(len(cand_list), K)

        if self.verbose and n_combos > 1000:
            print(f"      Enumerating {n_combos} subsets...")

        count = 0
        for subset in combinations(cand_list, K):
            count += 1
            if self.verbose and n_combos > 1000 and count % 1000 == 0:
                print(f"        {count}/{n_combos}...")

            x, obj, feasible = self._solve_on_support(c, A, b, set(subset))
            if feasible and obj < best_obj - 1e-12:
                best_obj = obj
                best_x = x

                # 和下界比较: 若 gap 极小, 可提前停止
                if abs(best_obj - f_lower) < 1e-10:
                    break

        if best_x is None:
            best_x = np.zeros(len(c))
            best_obj = np.inf

        return {
            'x': best_x,
            'objective': best_obj,
            'sparsity': int(np.sum(best_x > 1e-10)),
            'method': 'enumerate_k_subsets',
            'subsets_evaluated': count,
        }

    def _ga_search(self, c, A, b, cand_list, f_lower):
        """0-1 GA 在候选池上搜索"""
        n = len(c)
        cand_arr = np.array(cand_list)
        pool_size = len(cand_arr)
        K = min(self.K, pool_size)

        # 小型 GA 参数
        pop_size = min(100, self._n_choose_k(pool_size, K))
        max_gen = min(100, pool_size * 2)
        elite = max(3, pop_size // 10)

        rng = np.random.RandomState(42)

        # 初始化种群
        pop = []
        for _ in range(pop_size):
            chosen = rng.choice(pool_size, K, replace=False)
            pop.append(set(int(cand_arr[i]) for i in chosen))

        # 评估
        fit = [np.inf] * pop_size
        sol = [None] * pop_size
        feas = [False] * pop_size
        for i, s in enumerate(pop):
            sol[i], fit[i], feas[i] = self._solve_on_support(c, A, b, s)

        best_x = None
        best_obj = np.inf
        for i in range(pop_size):
            if feas[i] and fit[i] < best_obj:
                best_obj = fit[i]
                best_x = sol[i]

        # 缓存
        cache = {}
        for i, s in enumerate(pop):
            cache[tuple(sorted(s))] = (sol[i], fit[i], feas[i])

        no_imp = 0

        for gen in range(max_gen):
            # 选择
            ranked = sorted(range(pop_size),
                            key=lambda i: (not feas[i], fit[i]))
            elites = [set(pop[i]) for i in ranked[:elite]]
            offspring = []

            while len(offspring) < pop_size - elite:
                # 锦标赛选择
                t1 = rng.choice(pop_size, 3, replace=False)
                p1 = min(t1, key=lambda i: (not feas[i], fit[i]))
                t2 = rng.choice(pop_size, 3, replace=False)
                p2 = min(t2, key=lambda i: (not feas[i], fit[i]))

                # 交叉: 取并集后随机选 K 个
                union = list(pop[p1] | pop[p2])
                if len(union) >= K:
                    child = set(int(v) for v in
                                rng.choice(union, K, replace=False))
                else:
                    child = set(int(v) for v in
                                rng.choice(cand_arr, K, replace=False))

                # 变异
                if rng.random() < 0.6:
                    child_list = list(child)
                    n_replace = max(1, rng.poisson(1.0))
                    nd = min(n_replace, len(child_list))
                    child -= set(rng.choice(child_list, nd, replace=False))
                    new_idx = rng.choice(pool_size, nd, replace=False)
                    child.update(int(cand_arr[i]) for i in new_idx
                                 if int(cand_arr[i]) not in child)

                offspring.append(child)

            pop = elites + offspring

            # 重新评估
            fit = [np.inf] * pop_size
            sol = [None] * pop_size
            feas = [False] * pop_size
            for i, s in enumerate(pop):
                key = tuple(sorted(s))
                if key in cache:
                    sol[i], fit[i], feas[i] = cache[key]
                else:
                    sol[i], fit[i], feas[i] = self._solve_on_support(c, A, b, s)
                    cache[key] = (sol[i], fit[i], feas[i])
                    if len(cache) > 5000:
                        for _ in range(200):
                            cache.pop(next(iter(cache)), None)

            improved = False
            for i in range(pop_size):
                if feas[i] and fit[i] < best_obj - 1e-12:
                    best_obj = fit[i]
                    best_x = sol[i]
                    improved = True
                    no_imp = 0

            if not improved:
                no_imp += 1

            if no_imp > 30:
                break

        if best_x is None:
            best_x = np.zeros(n)
            best_obj = np.inf

        return {
            'x': best_x,
            'objective': best_obj,
            'sparsity': int(np.sum(best_x > 1e-10)),
            'method': f'ga_k{K}_pool{pool_size}',
            'generations': gen + 1,
            'lp_calls': len(cache),
        }

    def _greedy_search(self, c, A, b, cand_list):
        """贪心: 从免费LP支撑开始, 逐步替换"""
        n = len(c)
        K = self.K
        S_star = set(cand_list[:min(K, len(cand_list))])
        x, obj, feas = self._solve_on_support(c, A, b, S_star)

        if not feas:
            # 贪心构建: 从空集开始, 每次加最优变量
            S = set()
            for _ in range(K):
                best_j = None
                best_obj = np.inf
                for j in cand_list:
                    if j not in S:
                        test = S | {j}
                        _, o, f = self._solve_on_support(c, A, b, test)
                        if f and o < best_obj:
                            best_obj = o
                            best_j = j
                if best_j is not None:
                    S.add(best_j)
                else:
                    break
            x, obj, feas = self._solve_on_support(c, A, b, S)

        return {
            'x': x if x is not None else np.zeros(n),
            'objective': obj,
            'sparsity': int(np.sum(x > 1e-10)) if x is not None else 0,
            'method': 'greedy',
        }

    def _solve_on_support(self, c, A, b, support):
        """在给定支撑集上求解精确 LP"""
        arr = np.array(list(support))
        if len(arr) == 0:
            return None, np.inf, False

        try:
            c_sub = c[arr]
            if sparse.issparse(A):
                A_sub = A[:, arr].toarray()
            else:
                A_sub = A[:, arr]

            res = linprog(c_sub, A_ub=A_sub, b_ub=b,
                          A_eq=np.ones((1, len(arr))), b_eq=np.array([1.0]),
                          bounds=(0, None), method='highs')

            if res.success:
                x_full = np.zeros(len(c))
                x_full[arr] = res.x
                x_full[x_full < 1e-14] = 0
                return x_full, float(c @ x_full), True
        except:
            pass

        return None, np.inf, False

    @staticmethod
    def _n_choose_k(n, k):
        """计算 C(n,k) 的近似值 (避免溢出)"""
        if k > n:
            return float('inf')
        k = min(k, n - k)
        result = 1.0
        for i in range(k):
            result = result * (n - i) / (i + 1)
            if result > 1e9:
                return 1e9
        return int(result)


# ============================================================================
# 实验
# ============================================================================

def benchmark():
    """对比: 显式稀疏求解器 vs 免费LP vs 0-1GA vs ADMM"""
    from final_benchmark import generate_problem

    print("=" * 80)
    print("  显式 ||x||_0 <= K 稀疏LP求解器")
    print("=" * 80)

    configs = [
        (500, 10, 3),
        (1000, 10, 4),
        (2000, 10, 5),
    ]

    for n, m, K in configs:
        print(f"\n{'='*60}")
        print(f"n={n}, m={m}, K={K} (m+1={m+1})")
        print(f"{'='*60}")

        for trial in range(3):
            seed = 10000 + trial * 100
            prob = generate_problem(n=n, m=m, seed=seed)
            c, A, b = prob['c'], prob['A_csr'], prob['b']

            # 显式约束求解器
            solver = ExplicitSparseSolver(K=K, method='auto', verbose=True)
            result = solver.solve(c, A, b)

            # Gurobi (MIP formulation: x continuous, z binary indicator)
            try:
                import gurobipy as gp
                t0 = time.time()
                model = gp.Model("exp")
                model.setParam('OutputFlag', 0)
                model.setParam('Threads', 1)
                model.setParam('TimeLimit', 60)
                model.setParam('MIPGap', 1e-4)

                # x continuous in [0,1]
                xv = model.addMVar(n, lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS)
                # z binary: indicator of x_i > 0
                z = model.addMVar(n, vtype=gp.GRB.BINARY)
                # Big-M: x_i ≤ z_i (M=1 since x_i ≤ 1)
                model.addConstr(xv <= z)
                model.addConstr(z.sum() <= K)

                # LP constraints
                model.addConstr(xv.sum() == 1)
                model.addMConstr(A, xv, '<', b)

                model.setObjective(c @ xv, gp.GRB.MINIMIZE)
                model.optimize()

                t_mip = time.time() - t0
                if model.status in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT):
                    obj_mip = model.ObjVal
                    nnz_mip = int(np.sum(xv.X > 1e-8))
                else:
                    obj_mip = np.inf
                    nnz_mip = -1
                print(f"  Gurobi MIP: obj={obj_mip:.6f} nnz={nnz_mip} "
                      f"time={t_mip:.2f}s status={model.status}")
            except ImportError:
                obj_mip = np.inf
                nnz_mip = -1
                t_mip = np.inf

            print(f"  Gap from Gurobi MIP: "
                  f"{abs(result['objective']-obj_mip)/max(abs(obj_mip),1e-15)*100:.2e}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    benchmark()

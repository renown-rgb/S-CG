"""
================================================================================
最终对比实验: 稀疏单纯形LP (变量和为1)
三维度算法对比 vs Gurobi精确解

算法:
  A. 改进0-1 GA   — 智能初始化 + 稀疏交叉 + 自适应变异 + 局部搜索 + LP oracle
  B. 标准0-1 GA   — 随机初始化 + 均匀交叉 + 位翻转变异 + LP oracle (无改进)
  C. 标准GA(连续)  — 实数编码 + SBX + 修复算子 + 惩罚函数

维度: 100, 1000, 10000, 100000, 1000000
固定: m=10约束
================================================================================
"""
import numpy as np
from scipy.optimize import linprog
from scipy import sparse
import time, gc
from typing import Tuple, Optional, Set


# ============================================================================
# 0. 问题生成
# ============================================================================

def generate_problem(n: int, m: int = 10, seed: int = 42):
    rng = np.random.RandomState(seed)
    nnz_true = m + 1
    nz_idx = rng.choice(n, nnz_true, replace=False)
    x_true = np.zeros(n)
    x_true[nz_idx] = np.abs(rng.randn(nnz_true))
    x_true /= x_true.sum()

    density = 0.5 if n <= 1000 else (0.05 if n <= 10000 else 0.005)
    A = sparse.random(m, n, density=density, format='csr',
                      data_rvs=lambda sz: np.abs(rng.randn(sz)) * 0.5,
                      random_state=rng)
    b = A @ x_true + np.abs(rng.rand(m)) * 0.1 + 0.05
    c = rng.randn(n) * 2.0
    c[nz_idx] = np.abs(c[nz_idx])

    return {'c': c, 'A_csr': A, 'A_csc': A.tocsc(), 'b': b, 'n': n, 'm': m,
            'x_true': x_true, 'true_nnz': nnz_true}


# ============================================================================
# 1. 高效采样
# ============================================================================

def fast_sample(rng: np.random.RandomState, n: int, k: int,
                exclude: Set[int] = None) -> np.ndarray:
    k = max(1, min(k, n))
    result = set()
    while len(result) < k:
        vals = rng.randint(0, n, size=(k - len(result)) * 3)
        for v in vals:
            vi = int(v)
            if (exclude is None or vi not in exclude) and vi not in result:
                result.add(vi)
                if len(result) >= k: break
    return np.array(list(result))[:k]


# ============================================================================
# A. 改进0-1 GA
# ============================================================================

class Improved01GA:
    """改进0-1 GA — 智能初始化 + 稀疏交叉 + 自适应变异 + LS + 缓存"""

    def __init__(self, prob, pop_size=150, max_gen=200, K=12):
        self.c = prob['c']; self.A_csc = prob['A_csc']; self.b = prob['b']
        self.n = prob['n']; self.pop_size = pop_size; self.max_gen = max_gen
        self.K = K; self.elite = max(5, pop_size // 15)
        self.rng = np.random.RandomState()
        self._cache = {}; self._max_cache = 5000 if self.n < 100000 else 3000
        self._hits = 0; self._miss = 0
        self._sparse = self.n >= 50000

    def _lp(self, support):
        key = tuple(sorted(support))
        if key in self._cache: self._hits += 1; return self._cache[key]
        self._miss += 1
        if len(self._cache) >= self._max_cache:
            for _ in range(min(200, len(self._cache) // 2)):
                self._cache.pop(next(iter(self._cache)), None)
        arr = np.array(list(support))
        if len(arr) == 0: self._cache[key] = (None, np.inf, False); return self._cache[key]
        try:
            c_sub = self.c[arr]; A_sub = self.A_csc[:, arr].toarray()
            res = linprog(c_sub, A_ub=A_sub, b_ub=self.b,
                          A_eq=np.ones((1, len(arr))), b_eq=np.array([1.0]),
                          bounds=[(0, None)] * len(arr), method='highs')
            if res.success:
                xf = np.zeros(self.n); xf[arr] = res.x
                self._cache[key] = (xf, float(self.c @ xf), True)
            else: self._cache[key] = (None, np.inf, False)
        except: self._cache[key] = (None, np.inf, False)
        return self._cache[key]

    def _rand_support(self, k):
        arr = fast_sample(self.rng, self.n, k) if self._sparse \
              else self.rng.choice(self.n, min(k, self.n), replace=False)
        return set(int(v) for v in arr)

    def initialize(self):
        pop = []; K = self.K
        greedy = np.argsort(self.c)[:K * 3]
        for _ in range(self.pop_size // 4):
            chosen = self.rng.choice(greedy, min(K, len(greedy)), replace=False)
            pop.append(set(int(v) for v in chosen))
        for _ in range(self.pop_size // 4): pop.append(self._rand_support(K))
        for _ in range(self.pop_size // 4): pop.append(self._rand_support(max(1, K // 2)))
        for _ in range(self.pop_size - len(pop)): pop.append(self._rand_support(min(self.n, K * 3)))
        return pop[:self.pop_size]

    def _select(self, pop, fit, feas, k=3):
        idx = self.rng.choice(len(pop), k, replace=False)
        best = idx[0]
        for i in idx[1:]:
            if feas[i] and not feas[best]: best = i
            elif feas[i] == feas[best] and fit[i] < fit[best]: best = i
        return set(pop[best])

    def _crossover(self, s1, s2):
        union = list(s1 | s2)
        if not union: return {int(self.rng.randint(self.n))}
        k = max(1, min(len(union), int((len(s1) + len(s2)) / 2)))
        return set(int(v) for v in self.rng.choice(union, k, replace=False))

    def _mutate(self, s):
        child = set(s); r = self.rng.random()
        if r < 0.5:
            nd = min(self.rng.poisson(1.0), len(child))
            if nd > 0 and child:
                child -= set(int(v) for v in self.rng.choice(list(child), nd, replace=False))
            na = max(1, int(self.rng.poisson(1.5)))
            nv = fast_sample(self.rng, self.n, na, exclude=child) if self._sparse \
                 else self.rng.choice(self.n, min(na, self.n), replace=False)
            child.update(int(v) for v in nv[:na])
        elif r < 0.8:
            smp = fast_sample(self.rng, self.n, min(500, self.n), exclude=child) if self._sparse \
                  else self.rng.choice(self.n, min(500, self.n), replace=False)
            smp = np.array([v for v in smp if v not in child])[:min(100, len(smp))]
            if len(smp) > 0:
                best3 = smp[np.argsort(self.c[smp])[:min(3, len(smp))]]
                child.add(int(self.rng.choice(best3)))
        else:
            if len(child) > 1:
                e = list(child); child.remove(e[np.argmax(self.c[e])])
        if not child: child = {int(self.rng.randint(self.n))}
        return child

    def _ls(self, s):
        bs, bx, bo = set(s), None, np.inf
        x, obj, feas = self._lp(bs)
        if feas: bx, bo = x, obj
        smp = fast_sample(self.rng, self.n, min(150, self.n), exclude=bs) if self._sparse \
              else self.rng.choice(self.n, min(150, self.n), replace=False)
        for i in np.array([v for v in smp if v not in bs])[:30]:
            t = bs | {int(i)}; x, obj, feas = self._lp(t)
            if feas and obj < bo - 1e-12: bs, bx, bo = t, x, obj; break
        if len(bs) > 1:
            for i in sorted(bs, key=lambda v: self.c[v], reverse=True)[:20]:
                t = bs - {i}
                if not t: continue
                x, obj, feas = self._lp(t)
                if feas and obj < bo - 1e-12: bs, bx, bo = t, x, obj; break
        return bs, bx, bo

    def run(self):
        t0 = time.time()
        pop = self.initialize(); N = len(pop)
        fit = [np.inf] * N; sol = [None] * N; feas = [False] * N
        for i, s in enumerate(pop): sol[i], fit[i], feas[i] = self._lp(s)
        best_x = None; best_obj = np.inf
        for i in range(N):
            if feas[i] and fit[i] < best_obj: best_obj = fit[i]; best_x = sol[i]
        no_imp = 0
        for gen in range(self.max_gen):
            ranked = sorted(range(N), key=lambda i: (not feas[i], fit[i]))
            elites = [set(pop[i]) for i in ranked[:self.elite]]
            offspring = []
            while len(offspring) < N - self.elite:
                child = self._crossover(self._select(pop, fit, feas),
                                        self._select(pop, fit, feas))
                if self.rng.random() < 0.8: child = self._mutate(child)
                offspring.append(child)
            pop = elites + offspring
            fit = [np.inf] * N; sol = [None] * N; feas = [False] * N
            for i, s in enumerate(pop): sol[i], fit[i], feas[i] = self._lp(s)
            improved = False
            for i in range(N):
                if feas[i] and fit[i] < best_obj - 1e-12:
                    best_obj = fit[i]; best_x = sol[i]; improved = True; no_imp = 0
            if not improved: no_imp += 1
            if gen % 15 == 0 and gen > 5 and feas[0]:
                _, ls_x, ls_obj = self._ls(pop[0])
                if ls_obj < fit[0] - 1e-12:
                    sol[0] = ls_x; fit[0] = ls_obj; feas[0] = True
                    if ls_obj < best_obj: best_obj = ls_obj; best_x = ls_x; no_imp = 0
            if no_imp > 40: break
        if best_x is not None:
            x = best_x.copy(); x[x < 1e-14] = 0; x /= x.sum() if x.sum() > 0 else 1
        else: x = np.ones(self.n) / self.n
        return {'x': x, 'objective': float(self.c @ x),
                'sparsity': int(np.sum(x > 1e-14)), 'time': time.time() - t0,
                'lp_calls': self._miss, 'cache_hits': self._hits}


# ============================================================================
# B. 标准0-1 GA (无改进)
# ============================================================================

class Standard01GA:
    """
    标准二元GA — 0-1规划下的基础实现
    纯随机初始化 + 均匀交叉 + 固定位翻转 + 无LS + 无缓存
    评估同样用LP oracle (公平对比)
    """

    def __init__(self, prob, pop_size=150, max_gen=200, K=12):
        self.c = prob['c']; self.A_csc = prob['A_csc']; self.b = prob['b']
        self.n = prob['n']; self.pop_size = pop_size; self.max_gen = max_gen
        self.K = K; self.elite = max(3, pop_size // 20)
        self.rng = np.random.RandomState(789)
        self._lp_count = 0
        self._sparse = self.n >= 50000

    def _lp(self, support):
        self._lp_count += 1
        arr = np.array(list(support))
        if len(arr) == 0: return None, np.inf, False
        try:
            c_sub = self.c[arr]; A_sub = self.A_csc[:, arr].toarray()
            res = linprog(c_sub, A_ub=A_sub, b_ub=self.b,
                          A_eq=np.ones((1, len(arr))), b_eq=np.array([1.0]),
                          bounds=[(0, None)] * len(arr), method='highs')
            if res.success:
                xf = np.zeros(self.n); xf[arr] = res.x
                return xf, float(self.c @ xf), True
        except: pass
        return None, np.inf, False

    def _rand_support(self, k):
        k = max(1, min(k, self.n))
        if self._sparse:
            return set(int(v) for v in fast_sample(self.rng, self.n, k))
        return set(int(v) for v in self.rng.choice(self.n, k, replace=False))

    def _uniform_crossover(self, s1, s2):
        union = list(s1 | s2)
        if not union: return {self.rng.randint(self.n)}
        child = {v for v in union if self.rng.random() < 0.5}
        return child if child else {self.rng.choice(union)}

    def _bitflip(self, s):
        child = set(s)
        n_flip = max(1, self.rng.poisson(2.0))  # 固定期望2位翻转
        nd = min(n_flip // 2, len(child))
        if nd > 0 and child:
            child -= set(int(v) for v in self.rng.choice(list(child), nd, replace=False))
        na = max(1, n_flip - nd)
        if self._sparse:
            nv = fast_sample(self.rng, self.n, na, exclude=child)
        else:
            nv = self.rng.choice(self.n, min(na, self.n), replace=False)
        child.update(int(v) for v in nv[:na])
        return child if child else {self.rng.randint(self.n)}

    def _tournament(self, fit, feas, k=3):
        idx = self.rng.choice(len(fit), k, replace=False)
        best = idx[0]
        for i in idx[1:]:
            if feas[i] and not feas[best]: best = i
            elif feas[i] == feas[best] and fit[i] < fit[best]: best = i
        return best

    def run(self):
        t0 = time.time()
        N = self.pop_size
        pop = [self._rand_support(self.K) for _ in range(N)]
        fit = [np.inf] * N; sol = [None] * N; feas = [False] * N
        for i, s in enumerate(pop): sol[i], fit[i], feas[i] = self._lp(s)
        best_fit = min(fit); best_x = sol[np.argmin(fit)]
        for gen in range(self.max_gen):
            ranked = sorted(range(N), key=lambda i: (not feas[i], fit[i]))
            elites = [set(pop[i]) for i in ranked[:self.elite]]
            offspring = []
            while len(offspring) < N - self.elite:
                i1 = self._tournament(fit, feas); i2 = self._tournament(fit, feas)
                child = self._uniform_crossover(pop[i1], pop[i2])
                if self.rng.random() < 0.8: child = self._bitflip(child)
                offspring.append(child)
            pop = elites + offspring
            fit = [np.inf] * N; sol = [None] * N; feas = [False] * N
            for i, s in enumerate(pop): sol[i], fit[i], feas[i] = self._lp(s)
            bi = np.argmin([f if fe else np.inf for f, fe in zip(fit, feas)])
            if feas[bi] and fit[bi] < best_fit: best_fit = fit[bi]; best_x = sol[bi]
        x = best_x.copy() if best_x is not None else np.ones(self.n) / self.n
        x[x < 1e-14] = 0; x /= x.sum() if x.sum() > 0 else 1
        return {'x': x, 'objective': float(self.c @ x),
                'sparsity': int(np.sum(x > 1e-14)), 'time': time.time() - t0,
                'lp_calls': self._lp_count}


# ============================================================================
# C. 标准GA (连续编码)
# ============================================================================

class StandardContinuousGA:
    """实数编码 + SBX + 多项式变异 + 事后修复 + 惩罚"""

    def __init__(self, prob, pop_size=150, max_gen=200):
        self.c = prob['c']; self.A = prob['A_csr']; self.b = prob['b']
        self.n = prob['n']; self.pop_size = pop_size; self.max_gen = max_gen
        self.rng = np.random.RandomState(456)

    def _repair(self, x):
        x = np.clip(x, 0, None); s = x.sum()
        return x / s if s > 0 else np.ones(self.n) / self.n

    def _evaluate(self, x):
        obj = float(self.c @ x)
        cv = np.sum(np.maximum(0, self.A @ x - self.b) ** 2)
        return obj + 1e4 * cv

    def run(self):
        t0 = time.time(); n = self.n
        pop = [self._repair(np.abs(self.rng.randn(n))) for _ in range(self.pop_size)]
        fits = [self._evaluate(x) for x in pop]
        best_x = pop[np.argmin(fits)].copy(); best_f = min(fits)
        for gen in range(self.max_gen):
            ei = np.argsort(fits)[:max(2, self.pop_size // 15)]
            elites = [pop[i].copy() for i in ei]
            offspring = []
            while len(offspring) < self.pop_size - len(elites):
                i1, i2 = self.rng.choice(self.pop_size, 2)
                p1, p2 = pop[i1], pop[i2]; child = np.zeros(n)
                for i in range(n):
                    if self.rng.random() < 0.5:
                        u = self.rng.random()
                        if u <= 0.5: beta = (2 * u) ** (1 / 21)
                        else: beta = (1 / (2 * (1 - u))) ** (1 / 21)
                        child[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    else: child[i] = p1[i] if self.rng.random() < 0.5 else p2[i]
                if self.rng.random() < 0.2:
                    i = self.rng.randint(n); u = self.rng.random()
                    if u < 0.5: delta = (2 * u) ** (1 / 21) - 1
                    else: delta = 1 - (2 * (1 - u)) ** (1 / 21)
                    child[i] += delta * 0.1
                offspring.append(self._repair(child))
            pop = elites + offspring; fits = [self._evaluate(x) for x in pop]
            bi = np.argmin(fits)
            if fits[bi] < best_f: best_f = fits[bi]; best_x = pop[bi].copy()
        return {'x': best_x, 'objective': float(self.c @ best_x),
                'sparsity': int(np.sum(best_x > 1e-8)), 'time': time.time() - t0}


# ============================================================================
# D. Gurobi
# ============================================================================

def gurobi_solve(prob, timeout=120):
    import gurobipy as gp
    c = prob['c']; A = prob['A_csr']; b = prob['b']; n = prob['n']
    t0 = time.time()
    model = gp.Model("bench"); model.setParam('OutputFlag', 0)
    model.setParam('Method', -1); model.setParam('Threads', 1)
    model.setParam('TimeLimit', timeout)
    x = model.addMVar(n, lb=0.0, ub=1.0)
    model.addConstr(x.sum() == 1); model.addMConstr(A, x, '<', b)
    model.setObjective(c @ x, gp.GRB.MINIMIZE); model.optimize()
    t = time.time() - t0
    if model.status == gp.GRB.OPTIMAL:
        return x.X, model.ObjVal, int(np.sum(x.X > 1e-10)), t
    return None, np.inf, -1, t


# ============================================================================
# E. 实验
# ============================================================================

def run_benchmark():
    dimensions = [100, 1000, 10000, 100000, 1000000]
    m = 10
    all_results = []

    print("=" * 95)
    print("  多维度对比: 改进0-1GA | 标准0-1GA | 标准连续GA vs Gurobi")
    print(f"  n = {dimensions}, m={m}")
    print("=" * 95)

    for n in dimensions:
        seed = 500 + int(np.log10(max(n, 10))) * 100
        prob = generate_problem(n=n, m=m, seed=seed)
        K = m + 3

        if n <= 1000:      pop_sz, max_g = 120, 120
        elif n <= 10000:   pop_sz, max_g = 150, 180
        elif n <= 100000:  pop_sz, max_g = 180, 200
        else:              pop_sz, max_g = 200, 250
        sga_sz = min(pop_sz, 120 if n <= 100000 else 60)

        print(f"\n{'#'*60}")
        print(f"# n = {n:,} | pop={pop_sz} gen={max_g} | K={K}")
        print(f"{'#'*60}")

        # A. 改进0-1 GA
        print("  [改进0-1 GA] ...", end=" ", flush=True)
        t0 = time.time()
        iga = Improved01GA(prob, pop_size=pop_sz, max_gen=max_g, K=K)
        r_iga = iga.run()

        # B. 标准0-1 GA
        print(f"({r_iga['time']:.0f}s) | [标准0-1 GA] ...", end=" ", flush=True)
        s01 = Standard01GA(prob, pop_size=pop_sz, max_gen=max_g, K=K)
        r_s01 = s01.run()

        # C. 标准连续GA (n>=100K时跳过)
        skip_sga = (n >= 100000)
        if not skip_sga:
            print(f"({r_s01['time']:.0f}s) | [标准连续GA] ...", end=" ", flush=True)
            sga = StandardContinuousGA(prob, pop_size=sga_sz, max_gen=max_g)
            r_sga = sga.run()
            print(f"({r_sga['time']:.0f}s) | [Gurobi] ...", end=" ", flush=True)
        else:
            print(f"({r_s01['time']:.0f}s) | [连续GA SKIP] | [Gurobi] ...", end=" ", flush=True)
            r_sga = {'objective': np.inf, 'sparsity': -1, 'time': -1}

        # Gurobi
        x_g, obj_g, nnz_g, t_g = gurobi_solve(prob)
        print(f"({t_g:.0f}s)")

        # Gap计算
        gap_iga = abs(r_iga['objective'] - obj_g) / max(abs(obj_g), 1e-15) * 100
        gap_s01 = abs(r_s01['objective'] - obj_g) / max(abs(obj_g), 1e-15) * 100
        gap_sga = abs(r_sga['objective'] - obj_g) / max(abs(obj_g), 1e-15) * 100 if not skip_sga else np.inf

        all_results.append({
            'n': n, 'gurobi_obj': obj_g, 'gurobi_nnz': nnz_g, 'gurobi_time': t_g,
            'iga': r_iga, 's01': r_s01, 'sga': r_sga,
            'iga_gap': gap_iga, 's01_gap': gap_s01, 'sga_gap': gap_sga,
        })

        print(f"  Gurobi:      {obj_g:14.10f}  nnz={nnz_g:3d}  {t_g:6.1f}s")
        print(f"  改进0-1 GA:  {r_iga['objective']:14.10f}  nnz={r_iga['sparsity']:3d}  "
              f"gap={gap_iga:.2e}%  {r_iga['time']:6.1f}s  LP={r_iga.get('lp_calls','?'):,}")
        print(f"  标准0-1 GA:  {r_s01['objective']:14.10f}  nnz={r_s01['sparsity']:3d}  "
              f"gap={gap_s01:.2e}%  {r_s01['time']:6.1f}s  LP={r_s01.get('lp_calls','?'):,}")
        if skip_sga:
            print(f"  标准连续GA:  {'(SKIP - n too large)':>14s}")
        else:
            print(f"  标准连续GA:  {r_sga['objective']:14.10f}  nnz={r_sga['sparsity']:3d}  "
                  f"gap={gap_sga:.2e}%  {r_sga['time']:6.1f}s")
        gc.collect()

    # ========================
    # 汇总表
    # ========================
    print(f"\n\n{'='*95}")
    print("  汇总: 三维度算法 vs Gurobi")
    print(f"{'='*95}")

    # 表1: 目标值 + Gap
    print(f"\n  ┌─ 目标值与Gap ───────────────────────────────────────────────────────────────────────┐")
    print(f"  │ {'n':>8} │ {'Gurobi':>14} │ {'改进0-1GA':>14} {'Gap%':>10} │ {'标准0-1GA':>14} {'Gap%':>10} │ {'标准连续GA':>14} {'Gap%':>10} │")
    print(f"  ├──────────┼────────────────┼────────────────────────────┼────────────────────────────┼────────────────────────────┤")
    for r in all_results:
        print(f"  │ {r['n']:>8,} │ {r['gurobi_obj']:14.8f} │ {r['iga']['objective']:14.8f} {r['iga_gap']:10.2e} │ "
              f"{r['s01']['objective']:14.8f} {r['s01_gap']:10.2e} │ "
              f"{r['sga']['objective']:14.8f} {r['sga_gap']:10.2e} │")
    print(f"  └──────────┴────────────────┴────────────────────────────┴────────────────────────────┴────────────────────────────┘")

    # 表2: 稀疏度 + 时间
    print(f"\n  ┌─ 稀疏度与速度 ───────────────────────────────────────────────────────────────────────┐")
    print(f"  │ {'n':>8} │ {'Gurobi nnz':>10} {'t':>6} │ {'改进0-1GA nnz':>12} {'t':>6} │ {'标准0-1GA nnz':>12} {'t':>6} │ {'标准连续GA nnz':>12} {'t':>6} │")
    print(f"  ├──────────┼─────────────────┼────────────────────┼────────────────────┼─────────────────────┤")
    for r in all_results:
        print(f"  │ {r['n']:>8,} │ {r['gurobi_nnz']:>10} {r['gurobi_time']:6.1f} │ "
              f"{r['iga']['sparsity']:>12} {r['iga']['time']:6.1f} │ "
              f"{r['s01']['sparsity']:>12} {r['s01']['time']:6.1f} │ "
              f"{r['sga']['sparsity']:>12} {r['sga']['time']:6.1f} │")
    print(f"  └──────────┴─────────────────┴────────────────────┴────────────────────┴─────────────────────┘")

    # 表3: LP调用次数
    print(f"\n  ┌─ LP评估次数 ──────────────────────────────────────┐")
    print(f"  │ {'n':>8} │ {'改进0-1GA LP':>14} {'命中':>8} │ {'标准0-1GA LP':>14} │")
    print(f"  ├──────────┼────────────────────────┼──────────────────┤")
    for r in all_results:
        iga_lp = r['iga'].get('lp_calls', 0); iga_hit = r['iga'].get('cache_hits', 0)
        s01_lp = r['s01'].get('lp_calls', 0)
        print(f"  │ {r['n']:>8,} │ {iga_lp:>14,} {iga_hit:>8,} │ {s01_lp:>14,} │")
    print(f"  └──────────┴────────────────────────┴──────────────────┘")

    return all_results


if __name__ == "__main__":
    run_benchmark()

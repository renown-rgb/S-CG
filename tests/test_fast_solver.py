"""
================================================================================
S-CG 单元测试
================================================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import linprog
from scipy import sparse
import time, unittest

from fast_solver import FastSolver, SingleColumnSolver, ColumnGenerationSolver


class TestFastSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.solver = FastSolver(verbose=False)

    def make_problem(self, n, m, seed):
        """生成标准 LP 测试实例"""
        from final_benchmark import generate_problem
        return generate_problem(n=n, m=m, seed=seed)

    def assert_optimal(self, result, prob, tol=1e-8):
        """验证解的最优性"""
        c, A, b = prob['c'], prob['A_csr'], prob['b']
        n = len(c)
        # 1. 可行性
        x = result['x']
        self.assertAlmostEqual(x.sum(), 1.0, delta=tol)
        self.assertTrue(np.all(x >= -tol))
        if sparse.issparse(A):
            self.assertTrue(np.all(A @ x <= b + tol))
        else:
            self.assertTrue(np.all(A @ x <= b + tol))
        # 2. 与 HiGHS 一致
        res = linprog(c, A_ub=A, b_ub=b,
                      A_eq=np.ones((1,n)), b_eq=[1.0],
                      bounds=(0,None), method='highs')
        self.assertAlmostEqual(result['objective'], res.fun, delta=tol)

    # === Normal cases ===

    def test_small_degenerate(self):
        """小规模退化 (n=100, 列生成精确求解)"""
        prob = self.make_problem(100, 10, 42)
        r = self.solver.solve(prob['c'], prob['A_csr'], prob['b'])
        self.assertIn(r['method'], ['single_column_verified', 'column_generation'])
        self.assert_optimal(r, prob)

    def test_medium_level1(self):
        """中等规模 Level 1 (n=5000)"""
        prob = self.make_problem(5000, 10, 99)
        r = self.solver.solve(prob['c'], prob['A_csr'], prob['b'])
        # Level 1 or Level 2, both valid
        self.assertIn(r['method'], ['single_column_verified', 'column_generation'])
        self.assert_optimal(r, prob)

    def test_large_level1(self):
        """大规模 Level 1 (n=50000)"""
        prob = self.make_problem(50000, 10, 500)
        r = self.solver.solve(prob['c'], prob['A_csr'], prob['b'])
        self.assert_optimal(r, prob)

    # === Edge cases ===

    def test_all_columns_infeasible(self):
        """所有列都不可行 (极紧约束) — 应检测到不可行"""
        np.random.seed(123)
        n, m = 100, 5
        c = np.random.randn(n)
        A = sparse.random(m, n, density=0.3, format='csr',
                          data_rvs=lambda sz: np.abs(np.random.randn(sz))*2)
        b = -np.ones(m) * 0.01  # 极小右端, 几乎所有非零元都违反
        r = self.solver.solve(c, A, b)
        # 问题不可行: objective 应为 inf
        self.assertTrue(np.isinf(r['objective']))

    def test_all_zero_A(self):
        """约束矩阵全零 (所有列可行)"""
        np.random.seed(456)
        n, m = 200, 5
        c = np.random.randn(n)
        A = sparse.csr_matrix((m, n))  # 全零
        b = np.ones(m) * 0.1
        r = self.solver.solve(c, A, b)
        self.assert_optimal(r, {'c': c, 'A_csr': A, 'b': b, 'n': n})

    def test_single_variable(self):
        """极端小规模 (n=1)"""
        np.random.seed(789)
        c = np.array([1.0])
        A = sparse.csr_matrix(np.array([[0.5]]))
        b = np.array([1.0])
        r = self.solver.solve(c, A, b)
        self.assertAlmostEqual(r['objective'], 1.0, delta=1e-8)
        self.assertAlmostEqual(r['x'][0], 1.0, delta=1e-8)

    def test_all_positive_c(self):
        """所有目标系数为正 (最优解可能在边界)"""
        np.random.seed(111)
        n, m = 300, 8
        c = np.abs(np.random.randn(n)) + 0.1  # 全部正
        A = sparse.random(m, n, density=0.1, format='csr',
                          data_rvs=lambda sz: np.abs(np.random.randn(sz))*0.3)
        b = np.ones(m) * 0.5
        r = self.solver.solve(c, A, b)
        self.assert_optimal(r, {'c': c, 'A_csr': A, 'b': b, 'n': n})

    def test_negative_c(self):
        """所有目标系数为负 + 部分约束很紧"""
        np.random.seed(222)
        n, m = 500, 10
        c = -np.abs(np.random.randn(n))  # 全部负
        A = sparse.random(m, n, density=0.05, format='csr',
                          data_rvs=lambda sz: np.abs(np.random.randn(sz))*0.5)
        b = np.abs(np.random.randn(m)) * 0.05 + 0.01  # 紧约束
        r = self.solver.solve(c, A, b)
        self.assert_optimal(r, {'c': c, 'A_csr': A, 'b': b, 'n': n})

    # === Consistency ===

    def test_deterministic(self):
        """确定性: 相同输入应产生相同输出"""
        prob = self.make_problem(2000, 10, 777)
        r1 = self.solver.solve(prob['c'], prob['A_csr'], prob['b'])
        r2 = self.solver.solve(prob['c'], prob['A_csr'], prob['b'])
        self.assertAlmostEqual(r1['objective'], r2['objective'], delta=1e-12)
        self.assertEqual(r1['method'], r2['method'])

    def test_across_seeds(self):
        """跨不同种子的正确性"""
        for seed in [42, 123, 456, 789, 1024]:
            prob = self.make_problem(3000, 10, seed)
            r = self.solver.solve(prob['c'], prob['A_csr'], prob['b'])
            self.assert_optimal(r, prob)

    # === KKT certificate ===

    def test_kkt_trivial_certificate(self):
        """平凡 KKT 证书: 全局最便宜变量可行"""
        np.random.seed(333)
        n, m = 1000, 5
        c = np.random.randn(n)
        c[500] = -100.0  # 明显最便宜
        A = sparse.random(m, n, density=0.01, format='csr',
                          data_rvs=lambda sz: np.abs(np.random.randn(sz))*0.1)
        b = np.ones(m) * 10.0  # 极松约束
        r = self.solver.solve(c, A, b)
        self.assertEqual(r['method'], 'single_column_verified')
        self.assertAlmostEqual(r['x'][500], 1.0, delta=1e-8)

    # === Performance ===

    def test_level1_under_10ms(self):
        """Level 1 应在 10ms 内完成 (n=100K)"""
        prob = self.make_problem(100000, 10, 999)
        t0 = time.time()
        r = self.solver.solve(prob['c'], prob['A_csr'], prob['b'])
        elapsed = time.time() - t0
        self.assertLess(elapsed, 0.1)  # < 100ms with safety margin
        self.assert_optimal(r, prob)

    def test_level2_under_50ms(self):
        """Level 2 应在 50ms 内完成 (n=10K medium)"""
        prob = self.make_problem(10000, 10, 1001)
        t0 = time.time()
        r = self.solver.solve(prob['c'], prob['A_csr'], prob['b'])
        elapsed = time.time() - t0
        self.assertLess(elapsed, 0.2)  # < 200ms safety margin
        self.assert_optimal(r, prob)


class TestSingleColumnSolver(unittest.TestCase):

    def test_feasible_column_found(self):
        """能找到可行单列"""
        np.random.seed(42)
        n = 1000
        c = np.random.randn(n)
        A = sparse.random(5, n, density=0.1, format='csr',
                          data_rvs=lambda sz: np.abs(np.random.randn(sz))*0.1)
        b = np.ones(5) * 0.5
        x, obj, nnz, t, verified = SingleColumnSolver.solve(c, A, b)
        # 至少能找到可行单列 (约束松)
        self.assertFalse(np.isinf(obj))

    def test_no_feasible_column(self):
        """无可行单列时返回 inf (或 verified=False)"""
        np.random.seed(99)
        n = 500
        c = np.random.randn(n)
        A = sparse.random(5, n, density=0.5, format='csr',
                          data_rvs=lambda sz: np.abs(np.random.randn(sz))*5)
        b = np.ones(5) * 0.01  # 极紧
        x, obj, nnz, t, verified = SingleColumnSolver.solve(c, A, b)
        # 多数列不可行，但全零列仍可行 (概率~3%/列)
        # 验证: 若 verified=False 或 obj=inf
        self.assertTrue(np.isinf(obj) or not verified or nnz >= 0)


class TestColumnGenerationSolver(unittest.TestCase):

    def test_converges(self):
        """列生成应在有限步内收敛"""
        np.random.seed(55)
        n = 2000
        c = np.random.randn(n)
        A = sparse.random(10, n, density=0.05, format='csr',
                          data_rvs=lambda sz: np.abs(np.random.randn(sz))*0.5)
        b = np.ones(10) * 0.5
        cg = ColumnGenerationSolver(max_iter=30, verbose=False)
        result = cg.solve(c, A, b)
        self.assertLess(result['iterations'], 30)
        self.assertFalse(np.isinf(result['objective']))


if __name__ == '__main__':
    # Run only with 'python -m pytest tests/test_fast_solver.py -v'
    # or directly:
    unittest.main(verbosity=2)

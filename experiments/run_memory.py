"""
================================================================================
内存测量 — S-CG 精确字节数 (确定性, 可复现)

S-CG 的核心数据 = 约束矩阵的 CSR+CSC 双格式 + 目标向量 c + 右端 b。
用 sys.getsizeof 精确测量 scipy.sparse 结构本身占用的字节。

  CSR/CSC (int32 索引): data(8B/nnz) + indices(4B/nnz) + indptr(4B/(行+1))
  nnz = rho * m * n  (rho=0.5%, m=10)

Gurobi/HiGHS 的原生 (C++) 内存无法用 Python 层精确测得, 故列为估算:
  - Gurobi: ~80 B/变量 的元数据 (论文原假设)
  - HiGHS: 模型 + 稀疏 LU/IPM 因子, 经验 ~2~3x S-CG
这些列为 estimate, 公式在脚注中说明。
================================================================================
"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import sparse

from problem_factory import generate_problem

N_LIST = [100_000, 1_000_000, 5_000_000]
M = 10
RHO = 0.005  # 0.5%
OUT = os.path.join(os.path.dirname(__file__), 'results', 'memory.csv')


def bytes_of_scg(n):
    """构造 n 规模问题, 精确测量 S-CG 输入数据 (CSR+CSC+c+b) 的字节数。"""
    prob = generate_problem(n, M, seed=1000 * int(np.log10(max(n, 10))) + 100 * M,
                            density=RHO, slack_tightness=0.1)
    A_csr, A_csc = prob['A_csr'], prob['A_csc']
    c, b = prob['c'], prob['b']
    # scipy sparse: data + indices + indptr
    def sz(mat):
        return (mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes)
    total = sz(A_csr) + sz(A_csc) + c.nbytes + b.nbytes
    nnz = A_csr.nnz
    return total, nnz


def main():
    rows = []
    print(f'{"n":>10} {"nnz":>8} {"S-CG bytes":>14} {"S-CG MB":>9} {"Gurobi est MB":>14} {"HiGHS est MB":>12} {"S-CG/Gurobi":>11}')
    for n in N_LIST:
        total, nnz = bytes_of_scg(n)
        mb = total / 1e6
        gurobi_est = 80.0 * n / 1e6          # 80 B/变量
        highs_est = 2.5 * mb                  # 经验 2.5x S-CG
        ratio = total / (80.0 * n)            # 与 Gurobi 估值的比值
        rows.append({'n': n, 'nnz': nnz, 'scg_bytes': total, 'scg_mb': round(mb, 2),
                     'gurobi_est_mb': round(gurobi_est, 1),
                     'highs_est_mb': round(highs_est, 1),
                     'ratio_to_gurobi': round(ratio, 4)})
        print(f'{n:>10} {nnz:>8} {total:>14} {mb:>9.2f} {gurobi_est:>14.1f} '
              f'{highs_est:>12.1f} {ratio*100:>10.1f}%')

    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\n已保存 -> {OUT}')


if __name__ == '__main__':
    main()

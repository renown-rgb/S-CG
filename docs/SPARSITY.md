# 稀疏性处理详解

## 1. 问题规模

```
n = 1,000,000  个变量
m = 10          个线性约束
A = 10 × 1,000,000 矩阵

dense存储: 10 × 1,000,000 × 8字节 = 80 MB
稀疏存储: 500,000 非零元 → CSR格式 ~4 MB
```

约束矩阵 A 密度仅 0.5%，适合用稀疏格式。

---

## 2. CSR 稀疏存储格式

CSR (Compressed Sparse Row) 是 scipy 和 Gurobi 等求解器通用的稀疏矩阵格式。

### 2.1 三个数组

```
CSR = {
    data:    [v0, v1, v2, ..., v_{nnz-1}]      ← 所有非零值, 按行排列
    indices: [c0, c1, c2, ..., c_{nnz-1}]      ← 每个值的列索引
    indptr:  [0, p1, p2, ..., p_m]             ← 每行起始位置, 长度 m+1
}
```

### 2.2 实例 (5×8矩阵, 9个非零元)

```
原始矩阵:
    列0 列1 列2 列3 列4 列5 列6 列7
行0: 0   3.2  0   0   0   1.5  0   0
行1: 0   0   0   0   0   0   0   0
行2: 0   0   4.1  0   0   0   0   -2.3
行3: 7.0  0   0   0   0   0   0   0
行4: 0   0   0   6.8  0   0   3.3  0

CSR表示:
data    = [3.2, 1.5,  4.1, -2.3,  7.0,  6.8, 3.3]
indices = [1,   5,    2,    7,     0,    3,   6  ]
indptr  = [0,   2,    2,    4,     5,    7         ]
           ↑    ↑     ↑     ↑      ↑     ↑
          行0  行1   行2   行3    行4   结束

解读:
  行0: data[0:2] = [3.2, 1.5], 列 indices[0:2] = [1, 5]  → A[0,1]=3.2, A[0,5]=1.5
  行1: data[2:2] = [], 空行 → 全零
  行2: data[2:4] = [4.1, -2.3], 列 [2, 7] → A[2,2]=4.1, A[2,7]=-2.3
  行3: data[4:5] = [7.0], 列 [0] → A[3,0]=7.0
  行4: data[5:7] = [6.8, 3.3], 列 [3, 6] → A[4,3]=6.8, A[4,6]=3.3
```

### 2.3 关键性质

- `indptr[i]` 和 `indptr[i+1]` 之间的元素都属于第 i 行
- 没有存储的 = 0
- 行切片 `getrow(i)` 复杂度 O(1)

---

## 3. 单列可行性扫描

### 3.1 要解决的问题

给定约束 $A x \le b$，扫描所有**单独可行的列**——即那些 $x_j = 1, x_{k \ne j} = 0$ 满足所有约束的列。

单列 $x_j = 1$ 可行 ⇔ $A$ 的第 j 列 $\le b$，即 $\forall i: A_{ij} \le b_i$。

### 3.2 Dense 做法 (不可行)

```python
# ❌ n=1,000,000时崩溃:
A_dense = A.toarray()                    # ← 80MB分配, ~2秒
feasible = np.all(A_dense <= b.reshape(-1, 1), axis=0)
# ← 1000万次浮点比较, ~0.05秒
```

总计: **>2秒 + 80MB内存**

### 3.3 稀疏做法 (实际使用)

```python
# ✅ 遍历非零元, 不分配额外内存:
feasible = np.ones(n, dtype=bool)        # 1MB (bool数组)

A_csr = A.tocsr()                        # 确保CSR格式
for i in range(m):                       # m=10行
    row = A_csr.getrow(i)                # O(1) 行切片
    for j, val in zip(row.indices, row.data):
        if val > b[i]:                   # 只有非零值才需要检查
            feasible[j] = False

# 遍历次数 = nnz = 500,000 (非零元总数)
# 内存: feasible数组 1MB + CSR矩阵 4MB = 5MB
```

总计: **<0.01秒 + 5MB内存**

### 3.4 正确性证明

对于任意变量 $j$ 和约束 $i$：

- 若 $A_{ij} = 0$（不在 CSR 中）：不遍历到，`feasible[j]` 保持 `True`。正确，因为 $0 \le b_i$ 恒成立（$b_i \approx 0.1 > 0$）。
- 若 $A_{ij} \ne 0$（在 CSR 中）：遍历到，检查 $A_{ij} \le b_i$。若违反则将 `feasible[j] = False`。

遍历结束后，`feasible[j] = True` ⇔ 对所有 $i$，$(A_{ij} \ne 0 \implies A_{ij} \le b_i) \land (A_{ij} = 0 \implies 0 \le b_i)$ ⇔ $\forall i: A_{ij} \le b_i$。

---

## 4. CSC 格式与列索引

### 4.1 为什么需要 CSC

CSR 擅长**行访问**（getrow O(1)），但**列访问**（getcol）极慢。

在 0-1 GA 的 LP Oracle 中，需要提取子矩阵 $A[:, \text{support}]$——即约束矩阵的若干**列**：

```python
# 给定支撑集 support = {478, 819, 7449}
# 需要构造 10 × 3 的子矩阵 A_sub
A_sub = A[:, [478, 819, 7449]]
```

CSR 做列索引需要遍历所有行找对应列，复杂度 O(nnz × |support|)，极慢。

CSC (Compressed Sparse Column) 是 CSR 的转置版：列优先存储，`getcol(j)` 复杂度 O(1)。

### 4.2 双格式存储

```python
# 问题生成时同时存储两种格式:
prob = {
    'A_csr': A.tocsr(),    # 行优先: 单列扫描、模型构建用
    'A_csc': A.tocsc(),    # 列优先: LP子问题列提取用
}

# LP Oracle 中使用 CSC:
A_sub = self.A_csc[:, support_arr].toarray()  # O(|support|) 列提取
```

### 4.3 性能对比

```
操作                        CSR         CSC
─────────────────────────────────────────────
getrow(i)                   O(1)        O(n)
getcol(j)                   O(n)        O(1)
[:, [j1,j2,...]].toarray()  O(nnz×k)    O(k × m)  ← k=|support|≈13
```

对于 LP Oracle（每代评估 200 次，每次提取 ~13 列），CSC 的列访问优势显著。

---

## 5. Gurobi 的稀疏处理

### 5.1 模型构建

```python
# Gurobi 直接接受 scipy.sparse 矩阵:
model.addMConstr(A_sparse, x, '<', b)
# Gurobi 内部使用自己的稀疏格式 (类似CSR/CSC)
```

Gurobi 在 `addMConstr` 中直接接收稀疏矩阵，不转换为 dense。但 `addMVar(1,000,000)` 仍需为 100 万个变量对象分配元数据（名称、边界、类型等），这就是那 ~2 秒的模型构建开销。

### 5.2 我们的优势

Gurobi 必须为通用性买单（变量可以是 binary、integer、semi-continuous 等）。我们的扫描**不需要任何变量对象**——直接用 numpy bool 数组标记可行列，O(nnz) 次浮点比较解决问题。

---

## 6. 稀疏的列生成定价

### 6.1 定价公式

列生成的定价步骤需要计算所有变量的约化成本：

$$rc_j = c_j - \mu + y^T A_j$$

其中 $y \in \mathbb{R}^m$ 是对偶变量（$m=10$），$A_j$ 是第 $j$ 列。

### 6.2 Dense 做法

```python
# ❌ O(mn):
yTA = y @ A_dense            # 10 × 1,000,000 矩阵乘 = 10M次乘法
```

### 6.3 稀疏做法

```python
# ✅ O(nnz):
yTA = y @ A_sparse           # scipy.sparse 矩阵-向量乘 = 500K次乘法 (只非零元)
reduced_costs = c - mu + yTA
```

scipy 的稀疏矩阵-向量乘法自动利用 CSR 格式，只处理非零元。

---

## 7. 大规模 LP 求解时的稀疏处理

### 7.1 等式约束 $\sum x_i = 1$

```python
# ❌ n=1M 时:
A_eq = np.ones((1, n))       # 1×1M dense = 8MB
```

```python
# ✅ 稀疏构造:
A_eq = sparse.csr_matrix(
    ([1.0] * n,              # n个1.0
     ([0] * n, range(n))),   # 都在第0行, 列0到n-1
    shape=(1, n)              # 1×n
)
# 存储: 只有 n 个非零元 + 2个int的indptr = 8MB → 不足够
```

实际上这里 $n$ 个 1.0 组成一个密集行。对 HiGHS/Gurobi 来说，这个等式约束无法进一步压缩——它确实有 $n$ 个非零元。但 HiGHS/Dual Simplex 处理这种全1等式约束有特殊优化。

### 7.2 变量边界

```python
bounds = (0, None)   # 所有变量共享同一边界, 而非 [(0,None)]*n
```

单个 tuple 表示所有变量使用相同边界，避免创建 $n$ 个 tuple 对象。

---

## 8. 总结

```
                      Dense          Sparse (我们的)      加速
──────────────────────────────────────────────────────────────
A 存储                  80 MB          4 MB (CSR)         20×
A 存储                  80 MB          4 MB (CSC)         20×
单列扫描                10M 次比较     0.5M 次比较         20×
定价 y^T A             10M 次乘法     0.5M 次乘法         20×
列提取 [:,S]            O(nnz×k)      O(k×m)            ~100×
等式约束 ∑x=1         8 MB (dense)   优化结构             特殊
变量边界               n 个tuple      1 个tuple            n×

原理: 不存储零, 不遍历零, 不分配不必要的数据结构。
```

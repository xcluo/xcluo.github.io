- https://blog.51cto.com/u_16213652/12201210

### Sparse Matrix

```python
import scipy.sparse as sp

csr/c_matrix = sp.csr/c_matrix(matrix.numpy())
coo_matrix = sp.coo_matrix(matrix.numpy())

csr/c_matrix.indptr   # a1
csr/c_matrix.indices  # a2
csr/c_matrix.data     # a3

coo_matrix.row        # a1 
coo_matrix.col        # a2 
coo_matrix.data       # a3
```

#### COO
Coordinate List format 使用一个列表，把每个非零元素的行索引、列索引和值都枚举记录。

```
# Matrix.shape = (m, n)
1	0	2
0	0	3
4	5	6

# COO Matrix
a1 = Array(0, 0, 1, 2, 2, 2)
a2 = Array(0, 2, 2, 0, 1, 2)
a3 = Array(1, 2, 3, 4, 5, 6)
```

1. **Array 1** 存储非零元素行坐标信息
    - `len(a1) = #non-zero`
2. **Array 2** 存储非零元素纵坐标信息
    - `len(a2) = #non-zero`
3. **Array 3** 按行优先存储的非零元素数值
    - `len(a3) = #non-zero`

#### CSC
Compressed Sparse Column format 通过按列优先方式存储非零元素值及其位置信息来解决稀疏矩阵空间浪费问题，通常能将存储使用量降低多个数量级。CSC特别适合用于列操作（如快速访问一列）和基于矩阵的数学运算（如循环遍历非零元素实现矩阵-向量乘法）。

```
# Matrix.shape = (m, n)
1	0	2
0	0	3
4	5	6

# CSC Matrix
a1 = Array(0, 2, 3, 6)
a2 = Array(0, 2, 2, 0, 1, 2)
a3 = Array(1, 4, 5, 2, 3, 6)
```

1. **Array 1** 存储列非零元素个数信息
    - `len(a1) = n+1`，且`a1[0] = 0`，`a1[i]` 表示前`i`列 `a1[:,:i]` 中非零元素个数总数
2. **Array 2** 存储非零元素行坐标信息
    - `len(a2) = a1[-1]`，`a2[i]` 表示第`i`个非零元素的行坐标，可通过 `i` 与 `a1` 统计值的大小关系确定列坐标
3. **Array 3** 按列优先存储的非零元素数值
    - `len(a3) = a1[-1]`

#### CSR
Compressed Sparse Row format 通过按行优先方式存储非零元素值及其位置信息来解决稀疏矩阵空间浪费问题，通常能将存储使用量降低多个数量级。CSR特别适合用于行操作（如快速访问一行）和基于矩阵的数学运算（如循环遍历非零元素实现矩阵-向量乘法）。

```
# Matrix.shape = (m, n)
1	0	2
0	0	3
4	5	6

# CSR Matrix
a1 = Array(0, 2, 3, 6)
a2 = Array(0, 2, 2, 0, 1, 2)
a3 = Array(1, 2, 3, 4, 5, 6)
```

1. **Array 1** 存储行非零元素个数信息
    - `len(a1) = m+1`，且`a1[0] = 0`，`a1[i]` 表示前`i`行 `a1[:,:i]` 中非零元素个数总数
2. **Array 2** 存储非零元素列坐标信息
    - `len(a2) = a1[-1]`，`a2[i]` 表示第`i`个非零元素的列坐标，可通过 `i` 与 `a1` 统计值的大小关系确定行坐标
3. **Array 3** 按行优先存储的非零元素数值
    - `len(a3) = a1[-1]`


#### LIL
#### DOK
#### DIA

#### BSR
Block Sparse Row
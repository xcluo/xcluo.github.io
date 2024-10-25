#### SVD
奇异值分解Singular Value Decomposition，$A=USV^T\in\mathbb{R}^{m*n}$，其中

- $U\in\mathbb{R}^{m*m}$为方阵$AA^T$的特征矩阵，也叫左奇异向量矩阵；
- $S\in\mathbb{R}^{m*n}$为方阵$AA^T$或$A^TA$的非负（降序）奇异值的平方根矩阵。
- $V\in\mathbb{R}^{n*n}$为方阵$A^TA$的特征矩阵，也叫右奇异向量矩阵；

    > $A_{m*n}\approx U_{m*k}S_{k*k}V_{n*k}^T$即为压缩后的数据，此时存储值压缩为$k*(m+1+n)$个

    ```python
    import numpy
    from numpy import linalg as LA  # Linear Algebra
    U, Sigma, VT = LA.svd(mat)      # shape=【(m, m)】 【(min(m, n),)】 【(n, n)】
                                    # Sigma中奇异值降序排列

    def svd_dimension_reduce(U, s, VT, k):
        Sigma = np.zeros((mat.shape[0], mat.shape[1]))
        Sigma[:min(mat.shape[0], mat.shape[1]), :min(mat.shape[0], mat.shape[1])] = np.diag(s)
        mat_rank_k = U[:, :k] @ Sigma[:k, :k] @ VT[:k, :]
        return mat_rank_k
    ```
!!! info ""
    压缩：通过多个低维矩阵近似重构高维矩阵，特征数保持不变。需要计算协方差矩阵，计算量大
#### PCA
主成分分析Principal Component Analysis，即一组数据（m条数据）不同维度（n个维度）之间可能存在线性相关关系，PCA能够对这组数据（通过剔除协方差矩阵对应的小特征值维度）正交变换转化为各个维度之间（维度缩减为k）线性无关的数据，达到数据降维去噪的目的。

1. 零均值化处理，特征元素减去相应特征的均值$X_i=X_i-E[X_i] \in \mathbb{R}^{m*n}$
2. 计算协方差矩阵$C=X^TX\in\mathbb{R}^{n*n}$的特征值和特征向量
3. 按特征值的大小降序排列，选择对应的top-k个特征向量作为主成分得到矩阵$P\in\mathbb{R}^{n*k}$
    
    > k除了可以指定为具体的整数值外，还可以指定为百分数，对应满足≥k的特征值比重的最小k

4. 投影结果$Y=XP \in \mathbb{R}^{m*k}$即为降维到k维后的数据

    ```python
    from sklearn.decomposition import PCA

    pca = PCA(n_components=0.8)
    pca.fit(X)                  # 计算PCA投影矩阵，X为 [m, n] 数组
    Y = ret = pca.transform(X)  # 获取PCA投影结果，Y为 [m, k] 数组
    ```
!!! info ""
    降维：通过保留主要成分的投影结果，且特征数减少

#### t-SNE
t-distributed Stochastic Neighbor Embedding

!!! info ""
    降维：。需要迭代计算，计算时间长。
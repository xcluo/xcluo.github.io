Principal Component Analysis，即一组数据（m条数据）不同维度（n个维度）之间可能存在线性相关关系，PCA能够对这组数据正交变换（通过剔除协方差矩阵对应的小特征值维度）转化为各个维度之间（维度缩减为k）线性无关的数据，达到数据降维去噪的目的。

1. 零均值化处理，即每行元素减去当前行的均值$X_i=X_i-E[X_i] \in \mathbb{R}^{m*n}$
2. 求协方差矩阵$C=XX^T$
3. 对协方差矩阵$C$进行SVD奇异值分解，计算得到特征值和特征向量，其中特征值表示协方差矩阵特征的方差，特征向量表示线性变换的方向
4. 按照特征值的大小进行排序，选择对应的top-k个特征向量作为主成分得到矩阵$P\in\mathbb{R}^{n*k}$
    
    > k除了可以指定为具体的整数值外，还可以指定为百分数，对应满足≥k的特征值比重的最小k

5. $Y=XP \in \mathbb{R}^{m*k}$即为降维到k维后的数据

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.8)
pca.fit(X)                  # 计算PCA投影矩阵，X为 [m, n] 数组
Y = ret = pca.transform(X)  # 获取PCA投影结果，Y为 [m, k] 数组
```
### Matrix Factorization

#### SVD
1. 奇异值分解Singular Value Decomposition，$A=USV^T\in\mathbb{R}^{m*n}$，其中  

    - $U\in\mathbb{R}^{m*m}$为方阵$AA^T$的特征矩阵，也叫左奇异向量矩阵；
    - $S\in\mathbb{R}^{m*n}$为方阵$AA^T$或$A^TA$的奇异值平方根的降序非负矩阵。
    - $V\in\mathbb{R}^{n*n}$为方阵$A^TA$的特征矩阵，也叫右奇异向量矩阵；

2. Truncated Singular Value Decomposition，截断SVD  
    - $A_{m*n}\approx U_{m*k}S_{k*k}V_{n*k}^T$即为压缩后的数据，此时存储值压缩为$k*(m+1+n)$个

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
    SVD求特征向量和特征值时需要计算协方差矩阵，计算量较大

#### NMF
**非负矩阵**分解Non-negative Matrix Factorization，给定一个非负矩阵$V\in\mathbb{R}^{m*n}$，能够找到非负矩阵 $W\in\mathbb{R}^{m*k}$ 和 $H\in\mathbb{R}^{k*n}$，满足$V\approx WH$。

- $W\in\mathbb{R}^{m*k}$ features matrix特征矩阵，表示从原始矩阵中抽取出来的特征。**该部分可作为LSA主题模型结果**
- $H\in\mathbb{R}^{k*n}$ cofficients matrix系数矩阵，表示抽取出的特征与原有稀疏特征的关系。


NMF矩阵分解两种规优化目标及基于梯度下降的无监督迭代更新则如下：

1. Frobenius范数(矩阵L2范数): $\text{arg }\mathop{\text{min}}\limits_{W, H}\ \frac{1}{2}\Vert V-WH \Vert_F^2 = \frac{1}{2}\sum_{i, j}(V_{ij} - (WH)_{ij})^2$ 
2. KL散度: $\text{arg }\mathop{\text{min}}\limits_{W, H}\ D_{KL}(V\Vert WH) = \sum_{i, j} \big[V_{ij}\log \frac{V_{ij}}{(WH)_{ij}} - V_{ij} + (WH)_{ij} \big]$
    
    > $V, W, H$ 均为非负矩阵，所以可以进行KL散度计算  
    > $-V_{ij} + (WH)_{ij}$ 确保某一元素值为0时依然可以计算$V_{ij}$和$(WH)_{ij}$的差异

    ```python title="nmf"
    W, H = np.abs(np.random.rand(m, k)), np.abs(np.random.rand(k, n))
    for i in range(max_iter):
        # Frobenius范数更新规则
        W = W * ((V @ H.T) / (W @ H @ H.T + 1e-9))
        H = H * ((W.T @ V) / (W.T @ W @ H + 1e-9))
        error = np.linalg.norm(V - W@H)
        # KL散度更新规则，更新极慢
        V_over_WH = V / (W@H + 1e-9)
        W *= V_over_WH @ H.T / H.sum(axix=1)
        V_over_WH = V / (W@H + 1e-9)
        H *= (W.T @ V_over_WH) / W.sum(axis=0).T
        error = sum(V*np.log(V) - V*np.log(W@H)) -V + W@H 

        if error < tol:
            break
    return W, H
    ```
    > NMF分解矩阵更新出自论文: [Algorithms for Non-negative Matrix Factorization](https://proceedings.neurips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf)  
    > 迭代时可加入L1范式和L2范式进行正则规约，详见 `sklearn.decomposition.NMF`

#### LSA
潜在语义分析Latent Semantic Analysis，也被称为潜在语义索引（Latent Semantic Indexing，LSI），核心思想是利用矩阵分解技术来减少维度并发现文档与单词之间的潜在关系，从而克服了传统基于关键词的方法所面临的同义词和多义词问题

1. 使用Truncated SVD分解
    - $U\Sigma^{1/2}$ 表示文档在k维潜在语义空间的分布
    - $\Sigma^{1/2}V$ 表示k维潜在语义在词项空间的分布
   
2. 使用NMF分解
    - $W\in\mathbb{R}^{m*k}$ features matrix特征矩阵，表示从原始矩阵中抽取出来的特征。
    - $H\in\mathbb{R}^{k*n}$ cofficients matrix系数矩阵，表示抽取出的特征与原有稀疏特征的关系。

#### pLSA

#### LDA
Latent Dirichlet Allocation潜在狄利克雷分布，一种主体挖掘模型


- https://www.bilibili.com/video/BV123411G7Z9/?spm_id_from=333.337.search-card.all.click&vd_source=782e4c31fc5e63b7cb705fa371eeeb78
1. from document collection to get topics
2. LDA algorithm为几何分布, 接受率α取值{α=1, 位置均匀分布; α>1, 更倾向聚集在中心; α<1, 更倾向聚集在角落}  
    - 迪利克雷分布α和β，多项式分布表示分别为θ和φ，由θ生成的topics集合为Z，由φ生成的单词集合为W
    - $P(W, Z, \theta, \phi; \alpha, \beta)=\prod_{j=1}^MP(\theta_j; \alpha)\prod_{i=1}^K(\phi_i; \beta)\prod_{t=1}^{N_D}P(Z_{j,t}\vert \theta_j)P(W_{j,t}|\phi_{Z_{j,t}})$
    - $\prod_{j=1}^MP(\theta_j; \alpha)$: documents-topics DA
    - $\prod_{t=1}^{N_D}P(Z_{j,t}\vert \theta_j)$ distribution-topic 多项式分布，$N_D$ 由泊松分布采样得到
    - $\prod_{i=1}^K(\phi_i; \beta)$ 从β种得到一堆topics-words DA
    - $P(W_{j,t}|\phi_{Z_{j,t}})$ 给定topic情况下采样word，多项式分布
    - topic_number, hyperparameter
    - part 2 05:04, topic-words into document-topics
    1. from doc-topic 1xK distribution M1 generates sampling topic_k
    2. based on topic_k, from topic-word 1xN distribution M2 generates sampling word_n
    3. repeat 1,2 until generate N_d words
    
3. gibbs sampling吉布斯采样，通过某些对象的相对位置来对document-topic和topic-word整合着色
    - 每个doc的topic尽可能单色，每个word的topic尽可能单色，因此不会存在某个文档表现为多个(十个及以上)主题的现象
    - 因此会逐步将topic-word尽可能变为单色，最终表现document-topic为单色，实际上是最大或上述概率$P(W, Z, \theta, \phi; \alpha, \beta)$的一个过程
    - 利用条件分布来简化多维问题
    - LDA 并不是基于“完全接受”的原则工作的，而是依赖于先进的统计推断技术来处理其内部的概率结构。
    - https://www.jianshu.com/p/5c510694c07e  
    - from doc-words to get doc-topic and topic-words 


> LDA出自David M.Blei、吴恩达和Michael I.Jordan 2003年论文: [Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)


### Dimensionality Reduction
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
t-distributed Stochastic Neighbor Embedding t分布-随机邻近嵌入

!!! info ""
    降维：。需要迭代计算，计算时间长。
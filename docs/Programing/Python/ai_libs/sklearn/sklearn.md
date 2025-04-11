`pip install scikit-learn`

#### KMeans
```python
from sklearn.cluster import KMeans


class KMeans(_BaseKMeans):
    def __init__(
        self,
        n_clusters=8,           # 聚类簇数
        *,
        init="k-means++",       # 初始化质心方法
                                #   k-means++: 智能初始化质心，使初始质心彼此远离 
                                #   random: 随机初始化之心
        n_init="warn",
        max_iter=300,           # 最大迭代次数
        tol=1e-4,               # 收敛阈值
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",      # 更新算法
                                # lloyd: 标准EM风格算法
                                # elkan: 利用三角不等时加速
    ):
```

#### metrics

1. acc, f1, P, R, ROC_AUC
```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def accuracy_score(
    y_true,                 # ndarray, shape=(n, )
    y_pred,                 # ndarray, shape=(n, )
    *,
    normalize=True,         # {True: 返回acc, False: 返回 TP + TN}
    sample_weight=None      # Union[None, ndarray], 每个样本的权重
)


def f1/precision/recall_score(
    y_true,                 # ndarray, shape=(n, )
    y_pred,                 # ndarray, shape=(n, )
    *,
    labels=None,
    pos_label=1,            # 当average="binary"时，指定正样本的标签表示
    average="binary",       # {
                            #   'micro': 平均结果, 'macro': 加权平均结果, 
                            #   'samples', 'weighted', 
                            #   'binary': 报告pos_label的结果
                            # }
    sample_weight=None,
    zero_division="warn"
)
```
> 可使用 `#!python np.argmax(probs， axis=-1)` 获取 `#!python y_pred`
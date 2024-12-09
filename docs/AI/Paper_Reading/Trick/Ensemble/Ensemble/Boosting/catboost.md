### CatBoost
> 论文: CatBoost: unbiased boosting with categorical features  
> CatBoost: **Cat**egorical **Boost**ing  
> Yandex & Moscow Institute of Physics and Technology, NIPS 2017  


### 基本原理
- ordered boosting, a modification of standard gradient boosting algorithm to avoid target leakage  
- categrorical features processing
- categrorical features 是一个包含互相独立的离散特征集，常见的有one-hot encoding，对每个特征新增一个二分特征进行表示，但对于某些具有id属性的特征，二分属性往往不够，而是需要大量的枚举值作为区分
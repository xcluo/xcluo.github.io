```python
import random


## sample ##
"""
    - geometric_sample: 几何分布采样出n
    - random.uniform: [l, r] 之间均匀分布采样出v
"""
class random_sample:
    @staticmethod
    def geometric_sample(p, v=0, max_v=6):
        if v < 0 or p < 0 or p > 1:
            raise ValueError(f"v should ∈ [1, {max_v}], p should ∈ [0, 1]")
        if v >= max_v or random.uniform(0, 1) <= p:
            return v
        return random_sample.geometric_sample(p, v + 1, max_v)

    @staticmethod
    def uniform_sample(l=0, r=1):
        return random.uniform(l, r)

    
## choice ##
"""
    - random.choice: 等概率选取seq中的1个元素
    - random.choices: （指定各元素权重且）有放回地选取population中的k个元素
    - random.sample: 无放回地等概率选取population中的k个元素
"""
class random_choice:
    @staticmethod
    def random_choice(seq):
        return random.choice(seq)

    @staticmethod
    def random_choices(
            seq,
            weights=None,       # 每个元素被选中的相对权重，对应于population中的元素
            *,
            cum_weights=None,   # 截至idx-th元素的累计权重，对应于sum(weights[:idx])
            k=1                 # 指定抽样的元素个数
        ):
        return random.choices(seq, weights=weights, cum_weights=cum_weights, k=k)

    @staticmethod
    def random_sample(seq, k):
        return random.sample(seq, k)
```
import random


## sample ##
def geometric_sample(p, v=1, max_v=6):
    if v <= 0 or p < 0 or p > 1:
        raise ValueError(f"v should ∈ [1, {max_v}], p should ∈ [0, 1]")
    if v >= max_v or random.uniform(0, 1) <= p:
        return v
    return geometric_sample(p, v+1, max_v)


def uniform_sample(l=0, r=1):
    return random.uniform(l, r)


## choice ##

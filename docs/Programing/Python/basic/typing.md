---
title: "typing"
---

```python
from typing import Union, Optional
from typing import Callable, Dict, List, Tuple
from typing import Generic, TypeVar
```

```python
T = TypeVar("T")
class LXC(BaseModel, Generic[T]):
    a: str = ""
    b: T = None

    @classmethod
    def succ(cls, d):
        return LXC(b=d)

    @classmethod
    def bbb(cls):
        return LXC()

if __name__ == "__main__":
    d = "lxc"
    print(LXC.succ(d))
    print(LXC[list].succ([1, 2, 3]))    # 指定泛型数据类型再声明
```
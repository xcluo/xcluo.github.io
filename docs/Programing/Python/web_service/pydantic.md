---
title: "pydaqntic"
---


```python
from pydantic import BaseModel, Field
from pydantic import BeforeValidator, fieldevalidator


# 用于定义数据模型
class User(BaseModel):
    name: str: Field(default="John Doe", min_length=3)
    age: int


from typing import Annotated
```

```python
from typing import Annotated
# 自定义验证规则
Item = Annotated[str, BeforeValidator(lambda v: v.upper())]
```
> 更常用BaseModel
>
> - `BaseModel.model_validate(obj.__dict__)`

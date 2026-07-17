---
title: "Exception Handling"
---

### 执行逻辑

```python
try:
    # 可能引发异常的代码
    risky_operation()
except SomeException as e:
    # 当try块中发生SomeException异常时，执行此块
    print(f"捕获到异常: {e}")
else:
    # 仅当try块中没有发生任何异常时，执行此块
    print("操作成功，没有异常！")
finally:
    # 无论是否发生异常，都会执行此块，常用于清理资源
    print("此块总是会执行。")
```

!!! info
    若发生了异常，但不是 `SomeException`，会只执行finally部分

### Exception

=== "Exception"
    ```python
    class Exception(BaseException):
        def __init__(self, *args, **kwargs):
            ...

    e.args      # 输出args元组
    e.args[i+1] # 输出args元组第i+1个元素
    str(e)      # 等价于 ' : '.join(map(str, e.args))
    e.detail    # 输出kwargs中key为detail的值，不存在则报错
    ```

=== "自定义异常类"
    ```python
    class InvalidHeaderError(Exception):
        """
        An error getting jwt in header or jwt header information from a request
        """
        def __init__(self,status_code: int, message: str):
            self.status_code = status_code
            self.message = message
    ```
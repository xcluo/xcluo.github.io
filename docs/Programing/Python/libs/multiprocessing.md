
#### porcessinng.Process
```python
def __init__(
    self, 
    group=None,
    target=None,    # 调用的对象，进程执行的任务 
    name=None,      # 进程名字
    args=(),        # 调用对象的位置参数元组
    kwargs={},      # 调用对象的关键字参数字典
    *, 
    daemon=None     # bool, 是否为守护进程
    )
```

- `start`：启动进程，调用`run`方法
- `run`：进程运行的函数体
- `join(timeout=None)`：
- `terminate`：强制终止进程，不进行清理操作，且其子进程可能会变成僵尸进程
- `kill`：同`terminate`
- `close`：关闭进程对象，并清理资源，如果进程仍在运行则返回错误

#### processing.Pool

#### processing.Queue


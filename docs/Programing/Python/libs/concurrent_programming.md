


### threading
#### threading函数
```python
# 返回主线程
threading.main_thread()

# 枚举存活的线程 list[threading.Thread]
threading.enumerate()

# 返回存活的 threading.Thread 个数，等价于len(threading.enumerate())
threading.active_count()
```
> 兼容2.x版本（对应的方法使用的是驼峰命令法）

#### threading常量
```python
# 指定阻塞函数（如Lock.acquire()、RLock.acquire()、Condition.wait()等）中参数timeout
threading.TIMEOUT_MAX
```

#### `threading.Thread`
```python
def __init__(
    self, 
    group=None, 
    target=None,    # 调用的对象，线程执行的任务 
    name=None,      # 线程名字
    args=(),        # 调用对象的位置参数元组
    kwargs=None,    # 调用对象的关键字参数字典
    *, 
    daemon=None     # bool, 是否为守护线程
    )
```

- `start`：启动线程 (由于设置了flag, 该方法只能启动一次)，调用`run` 方法
- `run`：线程运行的函数体
- `join(timeout=None)` 以queue的形式让调用它的线程等待另一个线程运行结束后再执行（串行运行而不是并发运行）
- `name`：线程名
- `ident`：线程id
- `daemon`

### multiprocessing

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


### asyncio

### concurrent.features
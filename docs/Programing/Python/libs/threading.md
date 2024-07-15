
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
1. 方法
    - `run`：线程运行的函数体
    - `start`：启动线程 (由于设置了flag, 只能启动一次)
    - `join(timeout=None)` 以queue的形式让调用它的线程等待另一个线程运行结束后再执行（串行运行而不是并发运行）
    - `name`：线程名
    - `ident`：线程id
    - `daemon`
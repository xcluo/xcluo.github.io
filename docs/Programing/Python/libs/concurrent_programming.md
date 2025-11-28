
1. threading：多线程串行，可通过`global`直接共享全局变量
2. multiprocessing：多进程并行

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
- `run`：线程运行的函数体（运行传入的调用对象或继承类`Thread`重写方法run）
- `join(timeout=None)` 等待线程执行完再结束主线程
    - 一般统一`start`，再统一`join`，否则`start`和`join`穿插就是正常的串行
    ```python
    [t.start() for t in threads]
    [t.join() for t in threads]
    ```
- `name`：线程名
- `ident`：线程id
- `daemon`

#### `threading.Lock`
互斥锁被用来实现同时只有一个线程方寸共享资源，以避免无效赋值（如多个线程同时读取了赋值前的值）
```python
from threading import Lock
lock = Lock()

# 执行完pass_code将自动释放锁
while lock:
    pass_code
```
#### `threading.RLock`
#### `threading.Condition`
条件锁
#### `threading.Semaphore`
控制同时访问共享资源的线程数量
#### `threading.Event`
一个或多个线程需要知道另一个线程的状态才能进行下一步操作

- `is_set`：判断event是否enable
- `set`：置为enable
- `clear`：置为disable
- `wait(timeout=None)`：等待event状态为enable
1
```python
event = threading.Event()

def student_exam(student_id):
    print(f'学生{student_id}等监考老师发卷')
    event.wait()
    print(f'学生{student_id}开始考试了')

def invigilate_teacher():
    time.sleep(3)
    print('考试时间到了，学生们可以考试了')
    event.set()

def event_main():
    for student_id in range(3):
        threading.Thread(target=student_exam, args=(student_id+1,)).start()
    threading.Thread(target=invigilate_teacher).start()

event_main()        # 学生1等监考老师发卷
                    # 学生2等监考老师发卷
                    # 学生3等监考老师发卷
                    # 考试时间到了，学生们可以考试了
                    # 学生1开始考试了
                    # 学生3开始考试了
                    # 学生2开始考试了
```
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
- `join(timeout=None)`：等待进程执行完再结束主进程
    - 一般统一`start`，再统一`join`，否则`start`和`join`穿插就是正常的串行
    ```python
    [p.start() for p in processes]
    [p.join() for p in processes]
    ```
- `terminate`：强制终止进程，不进行清理操作，且其子进程可能会变成僵尸进程
- `kill`：同`terminate`
- `close`：关闭进程对象，并清理资源，如果进程仍在运行则返回错误

#### processing.Pool

#### processing.Queue



### concurrent.features
```python
from concurrent.futures import ThreadPoolExecutor, as_completed
```
```python
task_args = all_input_args_iterable

with ThreadPoolExecutor(max_workers=max_concurrent_workers) as executor:
    # 准备待提交的任务即相关输入参数，传参可单独指定，也可以*args直接解析
    future_to_tasks = [
        executor.submit(concurrent_function, *args) for args in task_args
    ]
    '''有索引异步任务
    future_to_tasks = {
        executor.submit(concurrent_function, *args): i for i, args in enmumerate(task_args)
    }
    for future in as_completed(future_to_tasks):
        idx = future_to_tasks[future]
    '''
    # as_completed 按任务完成的先后顺序来迭代处理结果
    for future in as_completed(future_to_tasks):
    # 按任务提交的先后顺序迭代处理结果，调用.result()方法可能会阻塞直至之前任务完成
    # for i, future in enumerate(future_to_tasks):
        try:
            # 获取异步任务concurrent_function返回结果
            result = feature.result()
        except Exception as e:
            print(f'任务执行出错：{e}')
```
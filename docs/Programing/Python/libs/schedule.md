---
title: "schedule"
---
```python
# pip install schedule
import schedule
```
schedule 是一个轻量级的 Python 定时任务调度库，提供了类似英语语法的 API，非常直观易用。

### 基本用法

#### Scheduler & Job
=== "自定义定时任务"
    ```python
    import schedule
    # 1. 使用默认schduler定义定时任务
    job = schedule.every(intercval=1)
    '''
    2. 设置interval或者指定具体时间，（详见Job）
    '''
    # 3. 加载定时任务
    job.do(func_name, *func_args, **func_kwargs)

    # 4. 通过默认scheduler调度执行定时任务
    while True:
        schedule.run_pending()
        time.sleep(1)
        # 5. 无调度任务时结束调度
        if not schedule.get_jobs():
            print("所有任务已完成")
            break
    ```

=== "Scheduler"
    ```python
    ```

=== "Job"
    ```python
    # 指定相对interval (interval可不为1)
    job.second/minute/hour/day/week
    job.seconds/minutes/hours/days/weeks

    # 指定绝对具体时间 (interval=1)
    job.day.at("10:30")                     # 每天10:30
    job.wednesday.at("13:15")               # 每周三15:15
    job.minute.at(":30")                    # 每分钟的第30s

    # 加载定时任务 (指定时间或interval返回值都为self, 即job == job.second)
    job.do(func_name, *func_args, **func_kwargs)
    ```

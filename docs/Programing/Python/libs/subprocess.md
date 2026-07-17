---
title: "subprocess"
---

subprocess 模块是 Python 标准库中用于创建和管理子进程的核心工具。它能从 Python 代码中启动新的进程，连接其输入/输出/错误管道，并获取返回码

### 常用方法

#### run

执行命令并等待其完成

```python
def run(*popenargs,             # *List[str] 要执行的命令
        input=None,             # 
        capture_output=False,   # 捕获子进程的 stdout 和 stderr 输出信息
        timeout=None,           # 设置命令执行的超时时间（秒），防止子进程无响应导致程序卡死
        check=False,            # 若子进程返回码非0，会抛出 CalledProcessError 异常
        **kwargs
    ):
```

#### Popen

---
title: "log"
---

## logging

```python
import logging
```

=== "getLogger"
    获取（或创建）Logger 实例。当调用 `getLogger(name)` 时，它会查找内部缓存中是否已有该名称的 Logger。
    - 存在 → 直接返回该实例。**同一名称在程序任何地方获得的都是同一个对象**
    - 不存在 → 创建一个新的 Logger 对象，注册到缓存，并返回
    ```python
    root_logger = logging.getLogger()
    app_logger = logging.getLogger('app')
    # 获取app子类module的Logger实例
    app_child_logger = logging.getLogger('app.module')

    # 常用方法
    logger = logging.getLogger(__name__)
    ```
=== "basicConfig"
    为根记录器（Root Logger）进行“一次性”的基础配置。
    ```python
    logging.basicConfig(
        filename=None,      # 日志文件路径
        filemode='a',       # 日志记录模式{a: 追加, w: 覆盖}
        level=None,         # 日志的最低记录级别
        format=None,        # 日志内容样式
        datefmt=None,       # 时间格式，如 "%Y-%m-%d %H:%M:%S"
        encoding=None       # 日志文件编码格式，推荐 "utf-8"
    )
    ```
=== "FileHandler"
    将日志记录持久化写入到磁盘文件中，方便事后排查和审计
    ```python
    logging.FileHandler(
        filename,
        mode='a',
        encoding=None       # 日志文件编码格式，推荐 "utf-8"
        delay=Fasle         # 是否仅在第一条日志写入时才创建文件
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    # TimedRotatingFileHandler
    # RotatingFileHandler
    ```
=== "StreamHandler"
    sss
=== "Formatter"
    sss

!!! info
    应用FileHandler完全可以不使用baseConfig

### 常用类方法

=== "Logger"
    ```python
    setLevel
    addHandler
    info
    ```
Formater

=== "FileHandler"
    ```python
    setFormatter
    setLevel
    ```

## loguru

```python
from loguru import logger
```

会自动输出日志的时间和日志代码所处文件的行

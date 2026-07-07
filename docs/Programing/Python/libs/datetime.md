---
title: "datetime"
---

datetime 模块是处理日期和时间的标准库

### 常用类

```python
from datetime import date, time, datetime, timedelta
from zoneinfo import ZoneInfo   # pip install tzdata
```

=== "date"
    date 类专门用于处理日期（年、月、日），是不可变（immutable）对象，所有操作都会返回新的 date 对象。
    ```python
    d1 = date(year, month, day)     # year, month, day 为对象属性
    d2 = date.today()
    d1.replace(year=None, month=None, day=None)
    d1.weekday()                    # {周一: 0, ..., 周日: 6}
    d1.isoweekday()                 # {周一: 1, ..., 周日: 7}
    ```

=== "time"
    time 类专门用来表示一天中的某个时刻（时、分、秒、微秒），是不可变（immutable）对象，所有操作都会返回新的
    ```python
    # hour, minute, second, microsecond 以及 tzinfo 为对象属性
    t = time(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    # 替换传入的非空字段
    t.replace(hour=None, minute=None, second=None, microsecond=None)
    ```

=== "datetime"
    datetime 类（继承了 `date` 类）可以看作是 date + time 的完整结合体，同时包含了年月日、时分秒和微秒信息，是日常开发中使用频率最高的类。它同样是不可变（immutable）对象。
    ```python
    dt = datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    dt = datetime.now(tz=None)                  # 获取指定时区的日期+时间
    dt = datetime.fromtimestamp(ts, tz=None)    # 将 Unix 时间戳（秒数）转为 datetime
    dt.replace(year=None, month=None, day=None, hour=None, minute=None, second=None, microsecond=None, tzinfo=True)
    ```

=== "timedelta"
    timedelta 代表两个日期或时间之间的时间差（持续时间），你可以把它理解为一个"时间段"或"时间增量"。比如 1天3小时、5分钟30秒 都是 timedelta。它同样是不可变（immutable）对象。
    ```python
    td = timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
    ```

=== "ZoneInfo"
    轻量级、高性能且完全符合 IANA 时区数据库标准的时区类
    ```python
    # 北京时间时区
    tz = ZoneInfo("Asia/Shanghai")
    ```

### 格式转化

strptime 和 strftime 主要用于在“字符串（String）”和“时间对象（datetime）”之间进行相互转换。

- **strptime** = string parse time（字符串 → 时间） `datetime.strptime(dt_str, fmt)`
- **strftime** = string format time（时间 → 字符串） `dt.strftime(fmt)`

常用格式化指令fmt如下

| 指令 | 含义 | 示例 |
| :--- | :--- | :--- |
| `%Y` | 四位数的年份 | 2024 |
| `%y` | 两位数的年份 | 24 |
| `%m` | 两位数的月份 | 05 |
| `%d` | 两位数的日期 | 17 |
| `%H` | 24小时制的小时 | 14 |
| `%I` | 12小时制的小时 | 02 |
| `%M` | 两位数的分钟 | 30 |
| `%S` | 两位数的秒 | 00 |
| `%f` | 微秒 | 000000 |
| `%Z` | 时区名称 | UTC, CST |

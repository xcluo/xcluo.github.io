---
title: colorama
---
```python
from colorama import init, Fore, Back, Style
init(autoreset=True)            # 自动重置
print(Fore.RED + "红色文字", "haha")
print(Back.RED + "红色背景")
print("hello world")
```
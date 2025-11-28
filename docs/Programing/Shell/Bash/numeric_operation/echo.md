---
title: "echo"
---

#### 基本语法
```bash
echo [options] [string]
```

- `-n` 不换行输出
- `-e` 启用反斜杠转义解释
- `-E` 关闭反斜杠转义解释e


#### 常用方法
```bash
# 单行输出多个字符串时，使用空格分隔
echo "Hello" "World" "!"    # hello world !
echo {1..5}                 # 1 2 3 4 5
```
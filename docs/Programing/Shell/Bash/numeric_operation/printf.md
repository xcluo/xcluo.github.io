---
title: "printf"
---

### 基本语法

```bash
printf format [arguments...]
```
#### format
format 格式为 `%[flags][width][.precision]specifier`

1. 标志 flags + 宽度 width + 精度precision
    - `-` 左对齐，`printf "%-10s" left`
    - `0` 不足用零填充，`printf "%05d" 42` # 输出 `00042`
    - `+` 显示正负号，`printf "%+d.5f" 42` # 输出 +42.00000
    - ` ` 用空格填充（足时也至少填充1），`printf "% d" 42` # 输出 " 42"

2. 类型指示符 specifier
    - `%d`, `%o`, `%x`, `%X`  将整数使用十进制，八进制，十六进制小写，十六进制大写表示
    - `%f` 浮点，常搭配指定精度 `.precision` 使用
    - `%s` 字符串
    - `%b` 反转义字符串 `printf "%b" "hello\nworld"` # 直接输入空格
    - `%q` 转化为Shell 可读的转义格式

### 常用方法
```bash
# PAD赋值
padded_num=$(printf "%010d" $num)`
# 多参数格式化输出
printf "Name: %-10s Age: %-10s Score: %-10s\n" "Name" "Age" "Score"               
```
> `printf` 不会自动换行
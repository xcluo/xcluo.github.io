---
title: "type"
---

## 基本语法

用于显示命令的类型信息，告诉用户给定的命令是别名、shell 函数、builtin 还是磁盘上的可执行文件等，基本语法为 `type [-afptP] NAME [NAME ...]`

Option

- `-t` 仅返回NAME对应的单词描述类型，如：`alias`、`builtin`、`function`、`file`、`keyword`
- `-a` 显示所有匹配项，包括 alias、builtin、function、file 等所有类型
- `-f` 在查找过程中跳过 Shell 函数
- `-P` 在 PATH 中查找并返回第一个匹配的磁盘文件路径，完全忽略别名、内建、函数等非文件类型。
- `-p` 只有当 NAME 被判定为外部磁盘文件（file 类型）时，才输出该文件的绝对路径；否则什么都不输出。

### 常用方法

```bash
# 查看命令类型
type ls          # ls is an alias

# 查看所有匹配项（常用）
type -a python   # 显示 python 的所有类型定义

# 仅返回类型名称
type -t cd       # builtin

# 查看命令的磁盘路径
type -p grep     # /usr/bin/grep
```

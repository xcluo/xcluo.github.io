---
title: "lsof"
---

lsof（**L**i**s**t **O**pen **F**iles）是一个强大的 Linux/Unix 命令行工具，用于列出当前系统上所有打开的文件。在 Linux 中，“一切皆文件”，因此 lsof 可以查看普通文件、目录、网络套接字、设备文件、管道等。

#### 基本语法

```bash
lsof [OPTIONS] [文件名|PID|用户]
```

Options

- `-i` 列出网络连接，可指定端口、协议
- `-u user_name` 指定用户打开的文件
- `-p pid` 指定进程打开的文件
- `-t` 只输出PID信息，方便传给其它命令

#### 常用命令

```bash
lsof -i TCP     # 列出TCP协议打开的文件
lsof -i :80     # 列出80端口打开的文件
lsof -t -i :80  # 列出80端口打开文件的PID（仅保留PID信息）
lsof -u xcluo   # 列出xcluo用户打开的文件
lsof -p 1       # 列出进程1打开的文件
```

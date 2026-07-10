---
title: "netstat"
---

**net**wokr **stat**istics 是经典的网络状态查看与统计工具，用于列出系统当前的网络连接、端口监听、路由表、网络接口统计等信息，是排查网络故障、查看端口占用的核心工具。

### 工具安装

`sudo apt install net-tools -y`

### 基本语法

1. `netstat [-vWeenNcCF]`
2. `netstat [-vWnNcaeol]`

Options:

- `-a` --all 所有socket
- `-l` --listening 所有监听的socket

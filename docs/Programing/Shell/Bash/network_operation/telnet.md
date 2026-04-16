---
title: "telnet"
---

telnet 是一款经典的远程登录与 TCP 端口测试工具，基于 Telnet 协议工作，常用于验证远程主机的特定端口是否可访问、简易远程登录（较少使用，安全性低）等。

### 工具安装
`sudo apt install telnet -y`

### 基本语法
`telnet [Options] [host [port]]`
> - host可以为ipv4、ipv6或域名
> - 若不指定端口号，则默认使用23号端口

Options:

- `-4` --ipv4，强制主机使用IPv4
- `-6` --ipv6，强制主机使用IPv6
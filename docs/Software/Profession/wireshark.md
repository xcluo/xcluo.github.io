---
title: Wireshark
---
免费抓包软件

## 基本语法

逻辑运算符， && || !

### options

#### ip

```
ip.src == 192.168.1.1       # 源ip
ip.dst == 192.168.1.1       # 目的ip
ip.addr == 192.168.1.1      # 源或目的ip任一匹配
ip.addr == 192.168.1.1      # 子网匹配（前24位匹配）
ip.addr >= 192.168.1.1      # 子网等于
ip.addr in {192.168.1.1}    # 属于
```

#### url

```
tcp.port == 30004           # 端口号
http contains "qa/list"     # 包含字符串
dns.qry.name contains "qa"  # 域名包含指定字符串
```

## 设置

- 时间显示：`视图 → 时间显示格式`

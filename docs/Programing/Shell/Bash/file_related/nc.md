Netcat

```bash
# 接收
nc -q 10 -lp 5678 | tar x

# 传输
ip addr  # 查看接收方ip address
tar c <sended_file_name> | nc -q 10 <receiver_ip_address> 5678
```
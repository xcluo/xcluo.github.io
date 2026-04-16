
关闭端口进程

1. 以管理员身份运行 CMD  
2. 查找占用 8000 端口的进程 `netstat -ano | findstr :8000`  
3. 根据最后一列 PID 结束进程 `taskkill /PID <PID> /F`

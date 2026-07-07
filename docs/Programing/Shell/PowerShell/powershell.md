---
title: "PowerShell"
---

CMD的脚本文件是 .bat，而PowerShell的脚本文件是 .ps1。这两种脚本不能互相直接运行。

关闭端口进程

1. 以管理员身份运行 CMD  
2. 查找占用 8000 端口的进程 `netstat -ano | findstr :8000`  
3. 根据最后一列 PID 结束进程 `taskkill /PID <PID> /F`

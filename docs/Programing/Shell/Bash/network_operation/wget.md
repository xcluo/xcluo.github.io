---
title: "wget"
---

wget（**W**eb **Get**）是一个强大的非交互式网络下载工具，支持 HTTP、HTTPS、FTP 协议。它可以在用户退出终端后继续执行（后台运行），支持断点续传、递归下载、镜像网站、限速等高级功能，非常适合脚本和自动化任务。

#### 基本语法

```bash
wget [OPTIONS] [URL...]
```

Options:

- `-O file_name` --output-document，指定保存的文件名
- `-P directory_path` --directory-prefix，指定保存目录
- `-i file_name` --input-file 从指定文件读取多个URL进行下载
- `-c` --continue 断点续传（续传未完成的下载）
- `--progress dot/bar` 显示进度条类型
- `-q`, --quiet 安静模式下载，不输出信息

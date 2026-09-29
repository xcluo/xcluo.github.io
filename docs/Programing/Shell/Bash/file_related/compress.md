---
title: "文件压缩/解压"
---

## gzip / gunzip

### gzip

压缩文件为 `.gz` 格式，**默认压缩后不保留原文件**。

```bash
gzip [OPTION]... [FILE]...
```

| 选项 | 说明 |
| ---- | ---- |
| `-d` | 解压缩（等价于 `gunzip`） |
| `-k` --keep | 压缩后保留原文件 |
| `-r` | 递归压缩目录下所有文件 |
| `-v` | 显示压缩比信息 |
| `-1` ~ `-9` | 压缩级别（1 最快，9 最小） |
| `-c` | 输出到标准输出，保留原文件 |

```bash
gzip file.txt                   # 压缩为 file.txt.gz
gzip -9 file.txt                # 最高压缩比压缩
gzip -c file.txt > file.txt.gz  # 保留原文件
gzip -k file.txt                # 保留原文件
```

### gunzip

解压缩 `.gz` 文件，等价于 `gzip -d`，**默认解压后不保留原文件**。

```bash
gunzip [OPTION]... [FILE]...
```

| 选项 | 说明 |
| ---- | ---- |
| `-r` | 递归解压目录下所有 `.gz` 文件 |
| `-v` | 显示解压文件信息 |
| `-k` | 解压后保留压缩文件 |

```bash
gunzip file.txt.gz   # 解压 file.txt.gz
```

---

## zip / unzip

### zip

压缩为 `.zip` 格式。

```bash
zip [OPTION]... [zip_file] [file]...
```

| 选项 | 说明 |
| ---- | ---- |
| `-r` | 递归处理，压缩目录下所有文件 |
| `-q` | 静默模式，不显示执行过程 |
| `-v` | 显示执行过程 |
| `-d` | 从压缩包中删除指定文件 |
| `-e` | 加密压缩（会提示输入密码） |
| `-u` | 更新或追加文件到压缩包 |
| `-f` | 更新现有文件 |
| `-m` | 压缩后删除原文件 |
| `-o` | 以压缩包内最新文件的时间为准 |

```bash
zip archive.zip file1.txt file2.txt    # 压缩多个文件
zip -r archive.zip folder/             # 递归压缩目录
zip -e archive.zip file.txt            # 加密压缩
```

### unzip

解压缩 `.zip` 文件。

```bash
unzip [OPTION]... [zip_file]
```

| 选项 | 说明 |
| ---- | ---- |
| `-d <dir>` | 解压到指定目录 |
| `-x <file>` | 排除指定文件 |
| `-l` | 列出压缩包内容不解压 |
| `-o` | 直接覆盖，不提示 |

```bash
unzip archive.zip              # 解压到当前目录
unzip archive.zip -d output/   # 解压到指定目录
unzip archive.zip -x "*.git/*" # 排除匹配的文件
unzip -l archive.zip           # 查看压缩包内容
```

> 安装：`apt install unzip`

---

## tar

打包并压缩/解压，常用 `.tar.gz`（tgz）格式。

```bash
tar [OPTION]... [FILE]...
```

| 选项 | 说明 |
| ---- | ---- |
| `-c` | 创建打包文件（压缩） |
| `-x` | 解开打包文件（解压） |
| `-f <file>` | 指定压缩包文件名 |
| `-v` | 显示处理文件信息 |
| `-z` | 使用 gzip 压缩/解压 |
| `-J` | 使用 xz 压缩/解压 |
| `-C <dir>` | 切换到指定目录，再执行压缩/解压 |

```bash
# 压缩
tar -czvf archive.tar.gz file1.txt folder/
tar -cJvf archive.tar.xz file1.txt folder/

# 解压
tar -xzvf archive.tar.gz
tar -xzvf archive.tar.gz -C /target/dir/
```

> 记忆：`c` = create，`x` = extract，`z` = gzip，`v` = verbose，`f` = file

---

## rar / unrar

### rar

压缩为 `.rar` 格式。

```bash
rar [OPTION]... [rar_file] [file]...
```

| 选项 | 说明 |
| ---- | ---- |
| `a` | 添加文件到压缩包 |
| `u` | 更新压缩包内文件 |
| `d` | 删除压缩包内文件 |
| `x` | 带路径解压（解压后保留目录结构） |
| `e` | 解压到当前目录（不保留目录结构） |
| `l` / `t` | 列出压缩包内容 / 测试压缩包完整性 |
| `v` | 详细列出压缩包内容 |
| `k` | 锁定压缩包，防止修改 |
| `rr` | 添加恢复记录 |
| `s` | 转换为固实压缩 |
| `-p<password>` | 设置密码 |
| `-o+` | 覆盖已存在文件 |
| `-o-` | 不覆盖已存在文件 |

```bash
rar a archive.rar file1.txt folder/       # 压缩文件/目录
rar a -r archive.rar ./                   # 递归压缩当前目录
rar a -p archive.rar file.txt             # 加密压缩
rar a -rr archive.rar file.txt            # 添加恢复记录
```

### unrar

解压缩 `.rar` 文件。

```bash
unrar [OPTION]... [rar_file] [path]
```

| 选项 | 说明 |
| ---- | ---- |
| `x` | 带路径解压（保留目录结构） |
| `e` | 解压到当前目录（不保留目录结构） |
| `l` | 列出压缩包内容 |
| `t` | 测试压缩包完整性 |
| `p` | 输出文件内容到标准输出 |
| `e+` | 解压并排除已存在文件 |
| `o+` / `o-` | 覆盖 / 不覆盖已存在文件 |
| `-p<password>` | 输入密码 |

```bash
unrar x archive.rar              # 带路径解压
unrar x archive.rar /target/dir/ # 解压到指定目录
unrar l archive.rar              # 列出压缩包内容
unrar t archive.rar              # 测试压缩包完整性
unrar e archive.rar              # 解压所有文件到当前目录
```

> 安装：`apt install unrar`（或 `apt install unrar-free`）

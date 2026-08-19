---
title: "make"
---

make 是 Linux/Unix 下的自动化构建工具，主要用于从源代码编译和安装软件。其标准编译安装流程为 `./configure`、`make`、`make install` 三步。

### ./configure 生成Makefile

配置脚本，用于检查系统环境并生成 Makefile。

| 选项 | 说明 |
| --- | --- |
| `--prefix=PATH` | 指定程序安装根目录（默认 `/usr/local`） |
| `--bindir=PATH` | 指定可执行文件目录 |
| `--sbindir=PATH` | 指定系统管理员可执行文件目录 |
| `--libdir=PATH` | 指定库文件目录 |
| `--includedir=PATH` | 指定头文件目录 |
| `--enable-feature` | 启用某功能 |
| `--disable-feature` | 禁用某功能 |
| `--with-package` | 依赖某软件包 |
| `--without-package` | 不依赖某软件包 |

> 常见用法：`./configure --prefix=/usr/local --disable-debug`

### make 编译

根据生成的 Makefile 进行编译。

| 选项 | 说明 |
| --- | --- |
| `-j[N]` | 并行编译，使用 N 个 jobs |
| `-B` | 无条件重新编译所有目标 |
| `-n` | 仅显示将要执行的命令，不实际执行 |
| `-C DIR` | 在执行前切换到指定目录 |
| `-f FILE` | 指定 Makefile 文件名 |

> 常用：`make -j$(nproc)` 利用所有 CPU 核心加速编译，一定要增加括号`()`

### make install 自动安装

将编译好的程序安装到系统目录。

| 选项 | 说明 |
| --- | --- |
| `DESTDIR=PATH` | 临时安装目录，用于打包（不污染系统） |

> 示例：`sudo make install` 或 `make DESTDIR=/tmp/package install`

### cp / chmod 手动安装

部分软件（如 `unrar`）编译后不提供 `make install`，需要手动拷贝可执行文件并设置权限。

```bash
# 解压后直接拷贝到系统目录
sudo cp unrar /usr/local/bin/
sudo chmod +x /usr/local/bin/unrar
```

常用安装路径：
- 可执行文件：`/usr/local/bin/`、`/usr/bin/`
- 库文件：`/usr/local/lib/`
- 配置文件：`/etc/`、`.conf` 文件通常放 `/usr/local/etc/`

### 完整示例

```bash
./configure --prefix=/usr/local
make -j$(nproc)
sudo make install
```

### 卸载

```bash
# 如果有 Makefile 且支持 uninstall
make uninstall

# 或者手动删除安装的文件（需要记录或查看 install_manifest）
```

### 清理

```bash
make clean    # 清除编译产生的目标文件
make distclean # 清除所有生成的文件（包括 Makefile）
```

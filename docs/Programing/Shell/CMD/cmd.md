---
title: "CMD"
---

CMD的脚本文件是 .bat，而PowerShell的脚本文件是 .ps1。这两种脚本不能互相直接运行。

## 常用命令

### 目录相关

#### `cd`

切换当前工作目录，基本语法为 `cd [/d] [drive:][path]`

```bash
cd /D E:\Projects   # 跨盘符切换目录
cd Videos           # 当前盘切换目录
```

#### `dir`

显示指定路径下所有文件和子目录的详细信息（包括文件名、扩展名、大小、上次修改的日期和时间等），基本语法为 `dir [<drive>:][<path>][<filename>] [OPTION]`

Option

- `/a`
- `/b` bare，仅列出文件名和扩展名，不包含大小、日期等详细信息
- `/w`
- `/o`

#### `md/mkdir`

创建新的目录/文件夹，基本语法为 `md [drive:]path`

#### `rd/rmdir`

删除目录/文件夹，基本语法为 `rd [OPTIONS] [drive:]path`

Option

- `/s` 删除文件夹本身及其内部所有子文件和子文件夹，类似于 `rm -rf`
- `/q` 安静模式，删除时不逐一确认，直接执行

### 文件相关

#### `del`

删除一个或多个文件，基本语法为 `del [OPTIONS] [drive:]path`

Option

- `/f` 强制删除只读文件
- `/s` 递归删除当前目录及所有子目录中匹配的文件
- `/q` 安静模式，不弹出确认提示
- `/p` 在删除每一个文件之前，都会提示用户进行确认

#### `type`

显示文本文件的内容，基本语法为 `type [drive:][path]<filename>`

### 进程相关

### 通用命令

#### `ren`

#### `findstr`

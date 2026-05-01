---
title: "os"
---

### os

```python
import os
```

#### 操作系统相关

```python
os.name                     # 返回操作系统类型，{posix: linux, nt: windows}
```

#### 环境变量

```bash
pip install python-dotenv
```

`.env` 文件是一个环境变量配置文件，用于存储应用程序的环境变量，通常位于项目根目录下。该方法具有以下优点：

1. **安全性**: 敏感信息不写入代码  
2. **环境隔离**: 不同环境使用不同配置  
3. **便捷性**: 配置集中管理，易于修改  
4. **团队协作**: 可以创建 .env.example 模板

```python
from dotenv import load_dotenv
import os

load_dotenv(
    dotenv_path=".env",         # 环境变量文件路径，可通过绝对路径指，如 `Path(__file__).parent / '.env'`
load_dotenv(dotenv_path=dotenv_path)
    override=False              # 是否覆盖加载
)     

# 访问环境变量，一般在配置类 `class Config()` 中读取
os.getenv(key)                  # 环境变量值都字符串类型（KEY为全大写）
```

#### 文件(夹)相关

=== "判断类"
    ```python
    os.path.exists(path)        # 是否存在文件(夹)
    os.path.isabs(path)         # 是否为绝对路径
    os.path.isfile(path)        # 是否为文件
    os.path.isdir(path)         # 是否为文件夹
    ```

=== "查看类"
    ```python
    os.getcwd()                 # 获取当前路径，get current work directory，等价于pwd
    os.path.abspath(path)       # 将路径转化为绝对路径
    os.path.relpath(path, 
                    start)      # 返回从起始路径start至目标路径path的相对路径

    os.listdir(path)            # 获取文件夹下所有文件，list[str]
    os.path.dirname(path)       # 获取路径的文件夹目录
    os.path.basename(path)      # 获取路径的文件名
    os.path.split(path)         # 等价于`(os.paht.dirname(path), os.path.basename(path))`
    os.path.splitext(path)      # 将文件拓展名.ext与文件名部分进行分割, tuple(path_to_name, .ext)
    
    os.walk(                    # 从top目录开始递归遍历目录树, 返回`generator(cur_dir, sub_dirs, sub_files)`
        top,                    # 根目录
        topdown=True,           # {True: 从父到子遍历，False: 从子到父遍历}
        onerror=None,           # 错误处理回调函数
        followliinks=False,
    )
    ```

=== "修改类"
    ```python
    os.mkdir(dir_name)          # 创建单层目录, 等价于mkdir
    os.makedirs(dirname/.../)   # 创建多层目录, 等价于mkdir -p
    os.rmdir  需要保证目录中无文件
    os.removedirs(dirname/.../)  # 需要保证目录中无文件
    // 删除文件
    os.remove(<file_name>)
    // 文件重命名
    os.rename(<file_name1>, <file_name2>)
    ```

### glob

```python
from glob import glob
glob("*.py")    # file list
```

### Path

```python
from pathlib import Path
```

=== "常用方法"
    ```python
    p = Path(__file__)           # 当前文件路径
    p = Path.cwd()               # 当前工作目录current work directory
    p = Path.home()              # 用户主目录
    p = Path() / "xcluo.py"      # Path()等价于Path("./")，使用 / 连算符拼接路径
    
    p.exists()                    # 是否存在文件(夹)
    p.is_file()                   # 是否为文件
    p.is_dir()                    # 是否为文件夹
    p.resolve()                   # 解析路径，返回绝对路径，会跳过符号链接
    ```

=== "常用属性"
    ```python
    p.name          # 含扩展文件名
    p.stem          # 不含拓展文件名
    p.suffix        # 扩展名，即p.split('.')[-1]
    p.suffixes      # 所有扩展名（如 .tar.gz 会返回 ['.tar', '.gz']）
    p.parent        # 父目录

    p.as_posix()    # 转换为正斜杠路径字符串
    p.as_uri()      # 转换为 file:// URI
    ```

=== "文件(夹)处理"
    ```python
    # 创建目录
    p.mkdir(
        parents=False,              # 是否同步创建父目录，类似于 `mkdir -p`
        exist_ok=False              # 是否忽略目录已存在报错
    )

    # 删除非空目录
    p.rmdir()

    # 读写文件（Path 对象自带！）
    p.write_text(
        "Hello, World!", 
        encoding="utf-8"
    )
    content = p.read_text(
        encoding="utf-8"
    )

    p.write_bytes(
        b"Hello, World!"
    )
    bytes_data = p.read_bytes()

    # 删除文件
    p.unlink(
        missing_ok=False            # 是否忽略缺失文件报错
    )
    ```

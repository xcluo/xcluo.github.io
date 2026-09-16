---
title: "PyInstaller"
---

# PyInstaller

将 Python 脚本打包成独立的可执行文件。

## 安装

```bash
pip install pyinstaller
```

## 基础用法

```bash
# 打包单个文件
pyinstaller --onefile myscript.py

# 打包目录（入口文件）
pyinstaller --onefile --name myapp myapp.py

# 打包成文件夹（不合并成单文件）
pyinstaller myscript.py
```

## 常用选项

| 选项 | 说明 |
|------|------|
| `-F` --onefile | 打包成单个可执行文件 |
| `-D` --onedir | 默认选项，打包成文件夹（多个文件），启动速度更快 |
| `-n package_name` --name | 指定输出可执行文件名称，默认为脚本名，名字含下划线时无效 |
| `-i icon_path` --icon | 指定程序图标，若更换icon时无效果，手动重命名生效（win缓存未释放导致） |
| `--clean` | 在开始构建前，清除全局缓存和上次构建留下的临时文件（build/） |
| `-w` | 使用窗口模式（无控制台） |
| `-c` | 使用控制台模式（默认） |
| `--debug` | 调试模式 |

## 配置文件

### .spec 文件

PyInstaller 会生成 `.spec` 文件，下次可直接修改并运行：

```python
# example.spec
a = Analysis(
    ['myscript.py'],
    pathex=[],
    binaries=[],
    datas=[('assets/*.png', 'assets')],  # 添加资源文件
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='myapp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon='icon.ico',
)
```

修改 spec 文件后运行：

```bash
pyinstaller myapp.spec
```

## 常见问题

### 隐藏导入

某些动态导入的模块需要手动指定：

```bash
pyinstaller --onefile --hidden-import=sklearn myscript.py
```

### 添加资源文件

```bash
pyinstaller --onefile --add-data "assets;assets" myscript.py
```

### 多个入口文件

```python
# spec 文件中修改
args=['script1.py', 'script2.py'],
```

### 打包后运行失败

1. 使用 `--debug=all` 查看错误信息
2. 检查是否有动态导入的模块需要 `--hidden-import`
3. 确认资源文件路径正确（打包后路径可能变化）

### 获取运行时路径

```python
import sys
import os

if getattr(sys, 'frozen', False):
    # 打包后的路径
    bundle_dir = sys._MEIPASS
else:
    # 开发环境的路径
    bundle_dir = os.path.dirname(os.path.abspath(__file__))

# 使用资源文件
resource_path = os.path.join(bundle_dir, 'assets', 'data.txt')
```

## 打包示例

```bash
# 完整示例：单文件、控制台、图标、自定义名称
pyinstaller --onefile --console --icon=app.ico --name myapp --add-data "resources;resources" main.py

# 无控制台（GUI程序）
pyinstaller --onefile --windowed --icon=app.ico --name myapp main.py
```

## 目录结构

打包完成后会生成：

```
dist/
  └── myapp.exe    # 最终可执行文件
build/              # 临时构建文件（可删除）
myapp.spec          # 配置文件
```
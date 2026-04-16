### 文本样式

1. 基础文本样式

    - **粗体**：`**粗体**`
    - *斜体*：`*粗体*`
    - ***斜体***：`***粗斜体***`
    - ==高亮==：`==高亮==`

2. 拓展文本样式（可在代码内部渲染使用）

    - {++绿色高亮 & 下划线++}：大括号+双加号
    - {--红色高亮 & 中划线--}：大括号+双减号
    - {~~红色 & 原内容~>绿色高亮 & 新内容~~}：大括号+双波浪线+中间用`->`分隔 原内容和新内容
    - {==高亮==}：大括号+双等号
    - {>>C语言多行注释<<}：大括号+双右/左定向符号
    - ++enter++：键盘符号，双加号+中间内容为键盘按键名小写

### 上/下标、划线

- ^上标^：`^上标^`
- ~下标~：`~下标~`
- ^^下划线^^：`^^下划线^^`
- ~~中划线（删除线）~~：`~~中划线~~`

### 链接与图片

- [链接文本](链接地址 "可选悬浮提示")：`[链接文本](链接地址 "可选悬浮提示")`
- ![MkDocs 图标](链接地址 "MkDocs 图标")：`![MkDocs 图标](链接地址 "MkDocs 图标")`


这是一段带有注释的文本[^1]。

[^1]: 这是注释的具体内容，会显示在文档底部。

# 1. 基础徽章
:octocat: （内置图标徽章）
:badge[最新版本] （普通文本徽章）
:badge[v1.0.0]{color=blue} （指定颜色徽章）

# 2. 链接徽章（带跳转）
:badge[MkDocs]{href=https://www.mkdocs.org/}


```python {linenos=table,hl_lines=2,4}
def hello():
    name = "MkDocs"  # 该行会高亮
    print("Hello")
    print(f"Hello {name}!")  # 该行会高亮
```

这是一段文本，带有脚注[1]。

[1]: 这是脚注内容，显示在文档底部。

# 使用 Markdown 扩展语法

这是一个功能 :material-check-circle:{ .mdx-badge } 已启用

<!-- PyPI 版本 -->
[![PyPI version](https://badge.fury.io/py/mkdocs-material.svg)](https://pypi.org/project/mkdocs-material/)

<!-- 构建状态 -->
[![Build Status](https://github.com/squidfunk/mkdocs-material/workflows/build/badge.svg)](https://github.com/squidfunk/mkdocs-material/actions)

<!-- Material for MkDocs 专用徽章 -->
[![Built with Material for MkDocs](https://img.shields.io/badge/Material_for_MkDocs-526CFE?style=for-the-badge&logo=MaterialForMkDocs&logoColor=white)](https://squidfunk.github.io/mkdocs-material/)

[version-badge]: https://img.shields.io/badge/版本-1.0.0-blue
[version-link]: https://你的项目地址

# 我的项目文档

## 项目信息

这是一个普通徽章：|"License":MIT|
这是一个GitHub安装徽章：|@github:six-two/mkdocs-badges|

# 简单徽章

|示例徽章|works|
|Python版本|3.10|

# 链接Link徽章

L|GitHub|https://github.com/six-two/mkdocs-badges|
L|文档|https://mkdocs-badges.six-two.dev/|

# 复制Copy徽章

C|密码|monkey123|
C|安装命令|pip install mkdocs-badges|
C|查看IP|ip a s eth0|

# 安装Install徽章

I|github|six-two/mkdocs-badges|
I|pypi|mkdocs|
I|npm|lodash|

# 标签Tag徽章

T|编程语言|Python|
T|难度|中级|


|x|9.7.0|class:version-badge|link:https://www.baidu.com|

- `copy:` 赋予标签复制属性
- `link:` 赋予标签链接跳转属性，url需要加上http(s)://
- `class:` 赋予标签样式属性


- 相邻徽章间距
- 徽章右对齐


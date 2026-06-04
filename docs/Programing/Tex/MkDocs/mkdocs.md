---
tags:                   # 分配 post 的标签
  - 博客
  - 教程
  - 入门
description: 学习如何使用 MkDocs 和 mkdocs-blog-plugin 创建自己的技术博客
---

#### 库相关

```bash
# base，mkdocs内置内容，应新尽新
pip install mkdocs-material     # (1)!

# 插件相关
pip install mkdocs-glightbox    # glightbox plugin，如图片放大、背景变暗突出图片以及自适应屏幕等交互响应式设计
pip install jieba               # 用于search plugin
pip install mkdocs-badges       # 用于徽章渲染
```

1.
    - uv环境中需要指定 `--livereload` 参数选项以实现动态加载；
    - `-a 0.0.0.0:PORT` 方式指定端口号

#### plugins

1. [`tags`](plugins/tags.md)

#### theme

1. [`comments`](theme/comments.md)

#### content

1. [`format`](content/format.md)
1. [`image`](content/image.md)
1. [`note`](content/note.md)
1. [`code`](content/code.md)
1. [`math`](content/math.md)
1. [`table`](content/table.md)
1. [`chart`](content/chart.md)

### MedaData

- https://zhuanlan.zhihu.com/p/613038183/

### 页面属性

https://squidfunk.github.io/mkdocs-material/setup/setting-up-tags/

#### 标题
执行网页标题

1. **使用**  
    - `title`：网页标题，纯str
    ```
    ---
    title: Hello_World
    ---
    ```

#### 页面成分显示

控制页面中的显示成分

1. **使用**
    - `hide`：需要隐藏的成分
    ```
    ---
    hide:
      - navigation # 隐藏左侧目录导航
      - toc        # 隐藏右侧页面导航
      - footer
    ---
    ```

#### 部件

#### 相对路径

```python
# in mkdocs.yml
use_directory_urls: false

# link resource
"can use relative path and absolute path"

# link file
"for .md: add file_name suffix"
"for .md head: file_name.suffix#head_name"
```

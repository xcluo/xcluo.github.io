### MedaData
---
slug: bigluo            # 别名，自定义post的文件名
title: luomou           # post的页面标题
categories:             # 分配 post 的类别
  - 教程
  - MkDocs
tags:                   # 分配 post 的标签
  - 博客
  - 教程
  - 入门
date: yyyy-MM-dd        # post创建日期
readtime: 15            # 手动指定post所需阅读时间，min
description: 学习如何使用 MkDocs 和 mkdocs-blog-plugin 创建自己的技术博客
---

- https://zhuanlan.zhihu.com/p/613038183/
### 页面属性
https://squidfunk.github.io/mkdocs-material/setup/setting-up-tags/

#### 博客
1. **激活**
    ```
    plugins:
        - blog  # 需提前 pip install mkdocs-blog-plugin
    ```

#### 标题
执行网页标题

1. **使用**  
    - `title`：网页标题，纯str
    ```
    ---
    title: Hello_World
    ---
    ```

#### 标签
指定网页标签信息

1. **激活**
    ```
    plugins:
        - tags
    ```

2. **使用**  
    - `tags`：网页标签，纯str list，可自定义标签
    ```
    ---
    tags:
      - HTML5
      - JavaScript
      - CSS
    ---
    ```


#### 评论显示
- `giscus`, `pip install mkdocs-giscus`
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


#### 主题


#### 部件
- [admonition](theme_related/admonition.md)


- [content tabs](https://squidfunk.github.io/mkdocs-material/reference/content-tabs/)
=== "C"
   
    ``` c
    #include <stdio.h>

    int main(void) {
      printf("Hello world!\n");
      return 0;
    }
    ```

=== "<span style="color: red">C++</span>"

    ``` c++
    #include <iostream>

    int main(void) {
      std::cout << "Hello world!" << std::endl;
      return 0;
    }
    ```

    ```python title="lxc"
    print("hello world")
    ```

### xxx
https://squidfunk.github.io/mkdocs-material/reference/code-blocks/

指定行内代码语言类型
```
`#!python range()`
```

```title="lxc"
print('hello world')
a = tf.zeros([3, 3], dtype=tf.float32)
```

- ==This was marked==
- a^T^，a~T~
- +++
- ^^This was inserted^^
- ~~This was deleted~~
- {deleted}
- Text can be {--deleted--} and replacement text {++ added ++}. This can also be
combined into {~~one~>a single~~} operation. {==Highlighting==} is also
possible {>>and comments can be added inline<<}.


``` yaml
# (1)!
```

1.  Look ma, less line noise!

```
| 左对齐 | 右对齐 | 居中对齐 |
| :-----| ----: | :----: |
| 单元格 | 单元格 | 单元格 |
| 单元格 | 单元格 | 单元格 |
```


| 左对齐 | 右对齐 | 居中对齐 |
| :-----| ----: | :----: |
| 单元格 | 单元格 | 单元格 |
| 单元格 | 单元格 | 单元格 |


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

#### 图片
```title="one-image"
<div class="one-image-container">
    <img src="image/FP32_demo.png" style="width: 80%;">
    <!-- <p>LoRA在Attention各部分权重上的消融实验效果</p> -->
    <!-- <figcaption>这是图片的标题或描述。</figcaption> -->
</div>
```

```title="row-image"
<div class="row-image-container">
    <div>
        <img src="image/FP32_demo.png" style="width: 80%;">
        <!-- <p>LoRA在Attention各部分权重上的消融实验效果</p> -->
        <!-- <figcaption>这是图片的标题或描述。</figcaption> -->
    </div>

    <div>
        <img src="image/FP32_demo.png" style="width: 80%;">
        <!-- <p>LoRA在Attention各部分权重上的消融实验效果</p> -->
        <!-- <figcaption>这是图片的标题或描述。</figcaption> -->
    </div>
</div>
```
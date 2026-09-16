---
date: 2024-01-31
pin: true     # 置顶推文 mkdocs-material>=9.7.0
slug: bigluo  # url 别名
title: "post 相关配置"
readtime: 15
comments: true
links:
  - AI/annotation.md
categories:
  - 入门
  - 博客
  - 教程
tags:
  - MkDocs
  - 博客配置
---



<!-- more -->
[Hover me](https://example.com "I'm a tooltip!")


[:material-pencil:]()

:fontawesome-brands-youtube:{ .youtube }

# Details
[1.9.0](https://example.com){ .md-badge .md-badge--primary}&nbsp;
[pymdownx.details](https://example.com){ .md-badge .md-badge--light }

Lorem ipsum[^1] dolor sit amet, consectetur adipiscing elit.[^2]


## 元数据
推文可以手动定义一组元数据属性，指定插件如何呈现它们、它们在何种视图中集成以及它们如何相互链接。每个帖子的元数据会根据模式进行验证，以便更快地发现语法错误。

- `pin: ture`
- `links:` 相对 docs 路径的文件链接，会在左侧相关链接栏显示
slug: 当前 post 的 url 别名


[^2]:
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

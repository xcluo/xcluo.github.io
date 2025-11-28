---
title: "ElasticSearch"
---

> Github: [elasticsearch](https://github.com/elastic/elasticsearch)

ElasticSerach是一个基于Lucene的搜索服务器，使用Java语言开发，基于RESTful web接口提供了一个分布式多用户能力的全文搜索引擎。官方客户端在Java、.NET（C#）、PHP、Python、Apache Groovy、Ruby和许多其他语言中都是可用的。

- [ElasticSearch Download](https://www.elastic.co/downloads/past-releases?product=elasticsearch)
    1. 下载指定版本，并解压
    2. 进入bin文件夹，并执行脚本`./elasticsearch.bat`
    3. 进入config文件夹，当`elasticsearch.yml`大小由2k → 5k时，修改配置如下
        ```
        xpack.security.enabled: false
        xpack.security.http.ssl:
        enabled: false
        ```
    4. 重新执行脚本 `./elasticsearch.bat`，浏览器数据`http://127.0.0.1:9200/` 查看是否启动成功
- 使用无模式的JSON（JavaScript对象表示法）文档存储数据
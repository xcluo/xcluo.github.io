---
title: "ElasticSearch"
---

- `pip install elasticsearch elasticsearch-dsl`
```python
from elasticsearch_dsl import connections
from elasticserach_dsl import Index, Document, Text, Integer, Boolean, Keyword
from elasticsearch_dsl import Q, Search, MultiSearch, Match, MultiMatch, MathchAll, MatchNone, MatchPhrase, MatchPhrasePrefix, MatchBoolPrefix
from elasticsearch_dsl.query import Regexp, Query, Fuzzy, Term
# regexp对输入条目是强制小写的，表达式大写无法匹配上
connection = connections.create_connection(
    alias="my_connection",
    hosts=["http://localhost:9200"],
    timeout=20
)

connection = connections.configure(
    default={"hosts", "localhost"},
    dev={
        "hosts": []
    }
)
```

- Index
```python
index = Index()             # 定义索引方式
index.create/exists/delete
```
- Document
- Kibana 是 Elastic Stack（原名 ELK Stack）的可视化平台，专门为 Elasticsearch 设计的前端界面。
    1. 下载并解压 (具体配置文件在`config/kibana.yml`)
    2. 进入bin文件夹，并执行脚本`./kibana.bat`
    3. 进入`127.0.0.1:5601` 查看是否启动成功

#### 分词 IK Analyzer
- 中文社区的事实标准，文档丰富，用户量大，稳定性非常高，经过多年大量生产环境验证。支持自定义词典，非常成熟，支持热更新。
- 下载放入`plugins`文件夹下，版本号要与ES版本一致
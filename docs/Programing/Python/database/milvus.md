---
title: "Milvus"
---

```python
# pip install pymilvus
from pymilvus import MilvusClient

# local
client = MilvusClient(
    uri="http://localhost:19530",   # 除了远程数据库，还可以为本地路径形式 milvus_demo.db
    user="",
    password="",
    db_name="",
    token="",                       # 验证密钥（如有需提供）
    timeout=None
    )

# collection → partition
# 向量集相关 # 
# 新建向量集
client.create_collection(
    collection_name,
    dimension=None,
    primary_field_name="id",        # 指定主键
    id_type="int"
    vector_field_name="vector",
    metric_type="COSINE",           # {IP: inner product, L2: euclidean distance, COSINE: cosine similarity}
    )

# 判断数据库中是否存在该向量集
client.has_collection(collection_name)

# C
client.insert(
    collection_name,
    data: : Union[Dict, List[Dict]],
    partition_name=""
    )
> # 元素除了id与vector字段外，还可以添加其他Collections Schema 中非定义字段，如text，它将自动添加到保留的 JSON 动态字段中，在高层次上可将其视为普通字段。在search时 output_fields可指定text字段返回
![alt text](image.png)

# R
res = client.search(
    collection_name,
    data: Union[List[list], list],  # vector or list[vector]
    filter="",
    limit=10,                       # top_k
    output_fields: Optional[List[str]]=None,  
                                    # 返回的字段
    partition_names: Optional[List[str]]=None,
                                    # 指定检索分区
    )
```

- [milvus混合检索](https://milvus.io/docs/zh/full_text_search_with_milvus.md)
- `schema = MilvusClient.create_schema()`，创建模式实现关键词索引
- `index_params = MilvusClient.prepare_index_params()` ，创建索引
- `client.hybrid_search()` ，混合检索
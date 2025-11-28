
## Gensim
```python
from gensim.models import KeyedVectors
```
Gensim （Generate Similar）核心目标是生成相似内容（如文档相似度计算、词向量建模等）。

### Embedding相关
#### KeyedVectors
类似于词向量的数据结构，用于存储词向量，提供相似度计算、词向量查找等功能。

1. 构建 `KeyedVectors`

    === "创建`KeyedVectors"
        ```python
        # 直接定义kv #
        kv = KeyedVectors(self,
            vector_size,        # 词向量维度
            count=0,            # 词向量行数,若指定将预分配空间,否则自动增加
        )

        # 从word2vec文件加载kv #
        kv = KeyedVectors.load_word2vec_format(cls,
            fname,              # word2vec file
            fvocab=None,
            binary=False,       # word2vec file是否为2进制格式
            encoding="utf-8",
        )        
        ```

    === "新增词向量"
        ```python
        kv.add_vector(
            key,                # word
            vector,             # vector
        )

        kv.add_vectors(
            keys,               # list[word]
            weights,            # 相应的list[vector]
            replace=False,      # 当key已存在时是否进行覆盖
        )
        ```


2. 相似度计算
    
    === "top-n获取"
        ```python
        # 获取给定key的top-n相似词 #
        kv.similar_by_key/word(
            key/word,               # word
            topn=10,                # 返回topn个相似词
            restrict_vocab=None     # Optional[int], 限定查询的词汇表大小
        ) -> list[(str, float)]

        # 获取给定vector的top-n相似词 #
        kv.similar_by_vector(
            vector,                 # vector
            topn=10,                # 返回topn个相似词
            restrict_vocab=None     # Optional[int], 限定查询的词汇表大小
        ) -> list[(str, float)]
        
        ```


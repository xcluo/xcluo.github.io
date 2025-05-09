
## Gensim
```python
from gensim.models import KeyedVectors
```
Gensim （Generate Similar）核心目标是生成相似内容（如文档相似度计算、词向量建模等）。

### Embedding相关
#### KeyedVectors

1. 加载模型 `KeyedVectors.load_word2vec_format`

    === "示例"
        ```python
        model = KeyedVectors.load_word2vec_format(r'E:\Python\projects\secession\data\label_data\v1/emojional.bin', binary=True)
        print(e2v)
        for key, value in model.vocab.items():
            print(key, value, type(value), value.index, e2v.vectors[value.index])
            break
        print(e2v.vectors[value.index].shape, len(e2v.vocab))
        ```
    === "定义"
        ```python
        def load_word2vec_format(cls, 
            fname, 
            fvocab=None, 
            binary=False, 
            encoding='utf8', 
            unicode_errors='strict',
            limit=None, 
            datatype=REAL
        )
        ```

2. 计算相似度 `most_similar`
    
    === "示例"
        sss

    === "定义"
        ```python
        def most_similar(self,
            positive=None, 
            negative=None, 
            topn=10, 
            restrict_vocab=None, 
            indexer=None)
        ```

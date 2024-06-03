

```python
from gensim.models import KeyedVectors
cus_vectors = KeyedVectors.load_word2vec_format(fname="E:/Embeddings/tencent-ailab-embedding-zh-d100-v0.2.0.txt")
cus_vectors.most_similar(positive=None, negative=None, topn=10, restrict_vocab=None, indexer=None)
```
1. `KeyedVectors.load_word2vec_format`
    ```python
    load_word2vec_format(cls, 
        fname, 
        fvocab=None, 
        binary=False, 
        encoding='utf8', 
        unicode_errors='strict',
        limit=None, 
        datatype=REAL
    )

    e2v = gensim.models.KeyedVectors.load_word2vec_format(r'E:\Python\projects\secession\data\label_data\v1/emojional.bin', binary=True)
    print(e2v)
    for key, value in e2v.vocab.items():
        print(key, value, type(value), value.index, e2v.vectors[value.index])
        break
    print(e2v.vectors[value.index].shape, len(e2v.vocab))
    ```

1. `cus_vectors.most_similar`
    ```python
    most_similar(self,
        positive=None, 
        negative=None, 
        topn=10, 
        restrict_vocab=None, 
        indexer=None)
    ```

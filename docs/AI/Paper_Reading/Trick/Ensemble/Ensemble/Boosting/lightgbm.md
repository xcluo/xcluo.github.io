`lightgbm==2.0.4`，为使dat文件可用

- 不再输出 `defualt_value`
- `lgb.train` 方法中删除`keep_training_booster=True`参数

### API

#### Data Structure API
1. Booster
2. Dataset

#### Training API

- train
- cv
```

```java
try {
    LightGBMModelConvertor.convert(
        txt_model_file,
        dat_model_file
    );
} catch (Exception e) {
    System.out.println("");
}
```
### Utils

#### tokenization.py

#### calculate_features.py

#### train_lightgbm.py
```python title="train_lightgbm.py"
import json
from scipy.sparse import csr_matrix
import random


def get_batch_samples(data, batch_size, shuffle=False, noleft=True):
    n = len(data)
    idxs = list(range(n))
    if shuffle:
        random.shuffle(idxs)

    start = 0
    for i in range(n // batch_size):
        yield [data[idx] for idx in idxs[start: start + batch_size]]
        start += batch_size

    if noleft and start < n:
        yield [data[idx] for idx in idxs[start:]]


class FeatureMap:

    def __init__(self,
                 feature_path,
                 feature_names):
        self.feature_path = feature_path
        self.feature_names = feature_names
        self.feature_word_score_index = self.read_features(feature_path, feature_names)

    def read_features(self):
        feature_word_score_index = []
        for feature_name in self.feature_names:
            word_score_index = dict()
            with open(self.feature_path + feature_name, 'r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)
                    word = line.get('w')
                    score = line.get('score')
                    index = line.get('id')
                    word_score_index[word] = (score, index)
            feature_word_score_index.append(word_score_index)
        return feature_word_score_index

    def build_sparse_vector(self, batch_tokens):
        batch_sparse_vector = []
        for tokens in batch_tokens:
            sparse_vector = [0.] * sum([len(feature_word_score_index) for feature_word_score_index in self.feature_word_score_index])

            prefix = 0
            for i in range(len(tokens)):
                for token in tokens[i]:
                    ret = self.feature_word_score_index[i].get(token, None)
                    if ret:
                        sparse_vector[ret[1] + prefix] = ret[0]
                prefix += len(self.feature_word_score_index[i])

            batch_sparse_vector.append(sparse_vector)
        return csr_matrix(batch_sparse_vector)
```
`lightgbm==2.0.4`，为使dat文件可用

- 不再输出 `defualt_value`
- `lgb.train` 方法中删除`keep_training_booster=True`参数

### API

#### Data Structure API
1. Booster
```python
bst = lightgbm.Booster(model_file=model_path + model_file_name)
x_tests = get_sparse_vector(get_by_index(ngrams, batch_idx), feats, n_feats)
preds = bst.predict(x_tests)
```
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

#### prechecker_tokenization
```python title="prechecker_tokenization.py"
from wheel_utils.char_alpha_numeric import PyTokenizer
import random
from tqdm import tqdm
import os
import json


def read_dataset(path, file_name):
    ret = []
    n0, n1 = 0, 0
    with open(path + file_name, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="reading total dataset"):
            line = json.loads(line)
            cnt = line.get('content')
            lbl = int(line.get('label', "0"))
            ret.append({'label': lbl, 'content': cnt})
            if lbl == 1:
                n1 += 1
            elif lbl == 0:
                n0 += 1
    print(f'in {file_name}:\n  #label_0: {n0}\n  #label_1: {n1}')
    return ret


def split_train_valid_dataset(dataset, tokenize_types, pp, path, train_percent=0.9, rewrite=False):
    """
        pp: 正样本增强倍数
        train_percent: 训练集种数据占所有数据的比例
    """
    if not rewrite and (os.path.exists(path + 'train.json') or os.path.exists(path + 'valid.json')):
        raise ValueError("train.json or valid.json already exist and rewrite is unavailable!!!")

    random.shuffle(dataset)
    n_train, n_valid = 0, 0
    with open(path + 'train.json', 'w', encoding='utf-8') as f_train, \
            open(path + 'valid.json', 'w', encoding='utf-8') as f_valid:
        for line in tqdm(dataset, desc="split train & valid dataset"):
            lbl = int(line['label'])
            cnt = line['content']
            if random.uniform(0, 1) < train_percent:
                ret = {'label': lbl}
                for tokenize_type in tokenize_types:
                    tokens = tokenize(cnt, tokenize_type)
                    ret[tokenize_type] = tokens
                for _ in range(1 if lbl == 0 else pp):
                    f_train.write(json.dumps(ret, ensure_ascii=False) + '\n')
                    f_train.flush()
                    n_train += 1
            else:
                ret = {'label': lbl}
                for tokenize_type in tokenize_types:
                    tokens = tokenize(cnt, tokenize_type)
                    ret[tokenize_type] = tokens
                f_valid.write(json.dumps(ret, ensure_ascii=False) + '\n')
                f_valid.flush()
                n_valid += 1
    print(f'using tokenize_types: {tokenize_types} and with positive sample augment of 【{pp}】, get'
          f'  \n#train_dataset: {n_train}  \n#valid_dataset: {n_valid}')


def tokenize(text, tokenize_type):
    if tokenize_type == 'char':
        tokens = char_tokenize(text)
    elif tokenize_type == 'sound':
        tokens = sound_tokenize(text, py_tokenizer)
    else:
        raise ValueError(f"{tokenize_type} is an invalid tokenize type!!!")
    return tokens


def char_tokenize(text):
    return list(text)


def sound_tokenize(text, py_tokenizer):
    sound = py_tokenizer.lazy_pinyin(text)
    return [c if not c.startswith('##') else c[2:] for c in sound]


if __name__ == "__main__":
    tokenize_types = ['char', 'sound']
    pp = 5  # 正样本增强
    train_percent = 0.9
    rewrite = False
    path = './../data/prechecker/dataset/'
    file_name = "data.json"
    sound_file = r'E:\JAVA\project_files\text-classification\textalg-check\src\main\resources\64_politics/char_meta.txt'

    py_tokenizer = PyTokenizer(
        sound_file,
        file_type="csv_1_2"
    )

    dataset = read_dataset(path, file_name)
    split_train_valid_dataset(
        dataset=dataset,
        tokenize_types=tokenize_types,
        pp=pp,
        path=path,
        train_percent=train_percent,
        rewrite=rewrite
    )
```

#### prechecker_features
```python title="prechecker_features.py"
import json
from scipy.sparse import csr_matrix
from math import log2
from tqdm import tqdm
from collections import Counter


def get_ngram(tokens, N):
    """
    get [1, N]-grams
    """
    ngrams = Counter(tokens)
    n = 2
    while n <= N and n <= len(tokens):
        tmps = ["".join(tokens[i:i + n]) for i in range(0, len(tokens) - n)]
        ngrams.update(tmps)
        n = n + 1
    return ngrams


def get_topk_features(batch_tokens, feature_top_k=20000, feature_min_freq=10):
    term_freqs = Counter()
    for item in batch_tokens:
        term_freqs.update(item)
    term_freqs = term_freqs.most_common(feature_top_k)
    vocab = {x[0] for x in term_freqs if x[1] > feature_min_freq}
    print("Get %d ngrams with min_freq=%d" % (len(vocab), feature_min_freq))
    return vocab


def chi_square(a, b, c, d):
    """
    chi_square = \frac{n*(ad - bc)^2}{(a+b)(c+d)(a+c)(b+d)}
        - |ad-bc|越小,特征和类别相关性越小
        - |ad-bc|越大,特征和类别相关性越大
    :param N11: a, 表示同时具有两种属性的个体数量
    :param N10: b, 表示具有第一个属性但不具有第二个属性的个体数量
    :param N01: c, 表示不具有第一个属性但具有第二个属性的个体数量
    :param N00: d, 表示同时不具有这两种属性的个体数量
    :return:
    """
    fenzi = (a + b + c + d) * (a * d - b * c) * (a * d - b * c)
    fenmu = (a + b) * (c + d) * (a + c) * (b + d)
    if fenmu == 0:
        return 0
    return fenzi * 1.0 / fenmu


def multual_infomation(a, b, c, d):
    """
    multual_infomation =\sum_{x \in X}\sum_{y \in Y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)}
    :param N_11: a, 表示同时具有两种属性的个体数量
    :param N_10: b, 表示具有第一个属性但不具有第二个属性的个体数量
    :param N_01: c, 表示不具有第一个属性但具有第二个属性的个体数量
    :param N_00: d, 表示同时不具有这两种属性的个体数量
    :return:
    """
    n = (a + b + c + d) * 1.
    I_UC = (a / n) * log2((a * n * 1.) / ((a + b) * (a + c))) + \
           (b / n) * log2((b * n * 1.) / ((a + b) * (b + d))) + \
           (c / n) * log2((c * n * 1.) / ((c + d) * (a + c))) + \
           (d / n) * log2((d * n * 1.) / ((c + d) * (b + d)))
    return I_UC


def jaccard(a, b, c, d):
    """
    jaccard = \frac{|A∩B|}{|A| + |B| - |A∩B|}
    :param N11: a, 表示同时具有两种属性的个体数量
    :param N10: b, 表示具有第一个属性但不具有第二个属性的个体数量
    :param N01: c, 表示不具有第一个属性但具有第二个属性的个体数量
    :param N00: d, 表示同时不具有这两种属性的个体数量
    """
    return a*1./(b + c)
    

def term_frequency(t_freq, doc_freq):
    return t_freq / (1. + doc_freq)


def select_features(batch_tokens, token2id, feature_type='chi'):
    if feature_type not in {'chi', 'freq', 'mi'}:
        raise ValueError('feature_type should in {"chi", "freq", "mi"}')

    n0 = len(batch_tokens[0])
    n1 = len(batch_tokens[1])
    feature = []

    if feature_type in ['chi', 'mi']:
        N = [[0, 0] for _ in range(len(token2id))]

        # count N_0x
        for tokens in tqdm(batch_tokens[0]):
            for token in tokens:
                ret = token2id.get(token, None)
                if ret:
                    N[ret][0] += 1

        # count N_1x
        for tokens in tqdm(batch_tokens[1]):
            for token in tokens:
                ret = token2id.get(token, None)
                if ret:
                    N[ret][1] += 1

        for token, idx in tqdm(token2id.items(), desc=f'calculate {feature_type}'):  # 是否包含词t / 是否属于目标类别.
            N_11 = N[idx][1]
            N_10 = N[idx][0]
            N_01 = n1 - N_11
            N_00 = n0 - N_10

            if N_00 * N_01 * N_10 * N_11 == 0:
                continue

            # 互信息计算
            if feature_type == "mi":
                metric_score = multual_infomation(N_11, N_10, N_01, N_00)
            # 卡方计算
            else:
                metric_score = chi_square(N_11, N_10, N_01, N_00)
            feature.append((token, metric_score))

    elif feature_type in ['freq']:
        doc_cnt = len(batch_tokens[1])
        xxx = [0] * len(token2id)
        for tokens in tqdm(batch_tokens[1]):
            for token in tokens:
                ret = token2id.get(token, None)
                if ret:
                    xxx[ret] += 1

        for token in tqdm(token2id, desc=f'calculate {feature_type}'):
            # C类文档集包含词项t的文档数
            t_doc_cnt = xxx[token2id[token]]
            metric_score = term_frequency(t_doc_cnt, doc_cnt)
            feature.append((token, metric_score))
        # feature.sort(key=lambda x: x[1], reverse=True)
    return feature


class FeatureMap:
    def __init__(self,
                 feature_path,
                 feature_names):
        self.feature_path = feature_path
        self.feature_names = feature_names
        self.feature_word_score_index = self.read_features()

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
            print(f'from {feature_name} reading {len(word_score_index)} features')
        return feature_word_score_index

    def build_sparse_vector(self, batch_tokens):
        batch_sparse_vector = []
        for tokens in batch_tokens:
            sparse_vector = [0.] * sum(
                [len(feature_word_score_index) for feature_word_score_index in self.feature_word_score_index])

            prefix = 0
            for i in range(len(tokens)):
                for token in tokens[i]:
                    ret = self.feature_word_score_index[i].get(token, None)
                    if ret:
                        sparse_vector[ret[1] + prefix] = ret[0]
                prefix += len(self.feature_word_score_index[i])

            batch_sparse_vector.append(sparse_vector)
        return csr_matrix(batch_sparse_vector)


def read_batch_tokens(path, file_name, tokenize_type, N):
    batch_tokens = [[], []]
    with open(path + file_name, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f'reading tokens file')):
            # {"label": lbl, feature_type: tokens ...}
            line = json.loads(line)
            lbl = line['label']
            tokens = line[tokenize_type]
            batch_tokens[lbl].append(get_ngram(tokens, N))

    # using positive samples to generate feature vocabulary
    vocab = get_topk_features(batch_tokens[1])
    token2id = {x: i for i, x in enumerate(vocab)}
    return batch_tokens, token2id


if __name__ == "__main__":
    feature_types = ['chi', 'freq', 'mi']
    tokenize_types = ['char', 'sound']
    N = 3
    top_k = 20000
    min_freq = 10
    feature_path = './../data/prechecker/features/'
    token_path = "./../data/prechecker/dataset/"
    token_file_name = 'train.json'

    for tokenize_type in tokenize_types:
        batch_tokens, token2id = read_batch_tokens(token_path, token_file_name, tokenize_type, N)

        for feature_type in feature_types:
            feature = select_features(batch_tokens, token2id, feature_type)
            with open(feature_path + f'{N}-gram_{tokenize_type}_{feature_type}.txt', 'w', encoding='utf-8') as f:
                for i, feat in enumerate(feature):
                    f.write(json.dumps({'id': i, 'score': feat[1], 'w': feat[0]},
                                       ensure_ascii=False) + '\n')
                    f.flush()
```

#### prechecker_lightgbm
```python title="prechecker_lightgbm.py"
import random
from prechecker_features import FeatureMap
import json
import lightgbm
import numpy as np
from tqdm import tqdm


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


if __name__ == "__main__":
    feature_types = ['chi', 'freq', 'mi']
    tokenize_types = ['char', 'sound']
    N = 3
    epochs = 5
    batch_size = 10000
    n_valid = 500
    learning_rate = 0.1

    param = {
        'objective': 'binary',
        'metric': 'binary',
        'learning_rate': 0.1,
        'boosting_type': 'gbdt',
        'feature_fraction': 0.9,        # feature dropout
        'bagging_fraction': 0.9,        # bagging dropout
        'bagging_freq': 5
    }

    feature_path = "./../data/prechecker/features/"
    token_path = "./../data/prechecker/dataset/"
    model_path = "./../data/prechecker/gbdts/"

    feature_map = FeatureMap(
        feature_path,
        [f'{N}-gram_{tokenize_type}_{feature_type}.txt'
         for feature_type in feature_types
         for tokenize_type in tokenize_types]
    )

    train_data_set = []
    with open(token_path + 'train.json', 'r', encoding='utf-8') as f_tok:
        for tok in tqdm(f_tok):
            tok = json.loads(tok)
            lbl = tok['label']
            tokens = [tok[tokenize_type] for tokenize_type in tokenize_types]
            train_data_set.append([lbl] + tokens)

    valid_data_set = []
    with open(token_path + 'valid.json', 'r', encoding='utf-8') as f_tok:
        for tok in tqdm(f_tok):
            tok = json.loads(tok)
            lbl = tok['label']
            tokens = [tok[tokenize_type] for tokenize_type in tokenize_types]
            valid_data_set.append([lbl] + tokens)

    model = None
    for epoch in range(epochs):
        print(f"############### EPOCH-{epoch+1} ###############")
        n_step = 1
        new_lr = learning_rate * (epochs - epoch) / epochs
        print("learning = " + str(new_lr))
        param["learning_rate"] = new_lr
        for batch_samples in get_batch_samples(train_data_set, batch_size, True):
            # 特征向量与标签向量.
            train_labels = [sample[0] for sample in batch_samples]
            train_tokens = [sample[1:] for sample in batch_samples]
            x_trains = feature_map.build_sparse_vector(train_tokens)
            y_trains = train_labels
            # 数据集.

            valid_samples = random.sample(valid_data_set, k=500)
            valid_tokens = [sample[1:] for sample in valid_samples]
            x_valids = feature_map.build_sparse_vector(valid_tokens)
            y_valids = [sample[0] for sample in valid_samples]

            train_data = lightgbm.Dataset(x_trains, label=np.array(y_trains))
            valid_data = lightgbm.Dataset(x_valids, label=np.array(y_valids), reference=train_data)
            n_step += 1

            model = lightgbm.train(
                param,
                train_data,
                num_boost_round=5,
                valid_sets=[valid_data],
                init_model=model)
            model.save_model(
                model_path + f"{N}-gram_{'-'.join(feature_types)}_model_{epoch+1}.txt")

```
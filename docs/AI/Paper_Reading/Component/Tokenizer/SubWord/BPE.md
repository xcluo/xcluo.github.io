[BPE (Byte Pair Encoding, 字节对编码)](https://www.derczynski.com/papers/archive/BPE_Gage.pdf) 是一种简单的数据压缩技术，它迭代地合并序列中最频繁的字节为单个未使用的字节。在分词任务中，合并的则是字符或字符序列。    

-  BPE是一个确定(无歧义)的、subwords替换word的贪心算法
- [Paper](media/pdf/BPE.pdf)

### 方法介绍
#### 基本原理
算法流程如下：

1. 设定字典中最大subwords个数|V|
1. 将所有单词拆分为subword序列，并在最后添加一个停止符`</w>`，同时标记出该单词出现的次数。例如，`"low"`这个单词出现了 5 次，那么它将会被处理为`{'l o w </w>': 5}`
1. 统计每一个连续字节对的出现频率，选择最高频者合并成新的subword
1. 重复第3步直到达到第1步设定的subwords词表大小|V|或下一个最高频的字节对出现频率为1

```python
{'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
```
最频繁的`subword_pair`是`e`和`s`，共出现 6+3=9 次，因此将它们合并
```python
{'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w es t </w>': 6, 'w i d es t </w>': 3}
```
最频繁的`subword_pair`是`es`和`t`，共出现 6+3=9 次，因此将它们合并
```python
{'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w est </w>': 6, 'w i d est </w>': 3}
```
最频繁的`subword_pair`是`est`和`</w>`，共出现了 6+3=9 次，因此将它们合并
```python
{'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w est</w>': 6, 'w i d est</w>': 3}
```
最频繁的`subword_pair`是`l`和`o`，共出现了 5+2=7 次，因此将它们合并
```python
{'lo w </w>': 5, 'lo w e r </w>': 2, 'n e w est</w>': 6, 'w i d est</w>': 3}
```
......持续迭代直到达到预设的subwords词表大小|V|或下一个最高频的字节对出现频率为1。


#### 代码实现
```python
import re, collections
def get_stats(vocab):
    '''
    # input
        vocab: {' '.join(subwords) + '</w>': freq},  (词的subwords list加上词终止符)
    # return
        pairs: {tuple_of_subword_pair: freq}
    '''
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        # 获取subwords
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs


def merge_vocab(best_pair, v_in):
    '''
    # input
        pair: {tuple_of_subword_pair: freq}
        v_in: {tuple_of_subword_pair: freq}
    # return
        v_out: {' '.join(subwords) + '</w>': freq}
    '''
    v_out = {}
    # 字符串转义表示
    bigram = re.escape(' '.join(best_pair))
    # 通过指定前后不是非空字符(\S为\s的补集, 包括^和$)约束只匹配subword_pair段
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')     
    for word in v_in:
        # 合并subword_pair并更新当前状态
        w_out = p.sub(''.join(best_pair), word)
        v_out[w_out] = v_in[word]
    return v_out


vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2,
         'n e w e s t </w>':6, 'w i d e s t </w>':3}
num_merges = 10
for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)    # 获取max_freq对应的tuple_of_subword_pair
    vocab = merge_vocab(best, vocab)
    print(best)
```

### 编码和解码
#### 编码

#### 解码

### 注意事项
#### 特殊编码示例
1. 部分贪心匹配 --> `<unk>`
    ```python
    vocab1 = {'①', '#②③'}
    vocab2 = {'①', '①②', '#②③'}
    print(tokenze('①②③', vocab1))   # ['①', '#②③']
    print(tokenze('①②③', vocab2))   # [<unk>]
    ```

#### BPE词表生成
- 训练集和测试集一起参与词表的生成  
> 保证词表一致性
- 翻译或生成任务中，不同语种的数据也可统一参与词表的生成  
> 避免一些专有名词在不同语种中划分为不同的subword序列
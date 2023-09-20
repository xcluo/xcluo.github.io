 SentencePiece实现了 [BPE](BPE.md) 和 [ULM](ULM.md) 两种subword算法，且能直接从原始文本语料训练得到subword词表。

- [Paper](media/pdf/SentencePiece.pdf)
- [Github](https://github.com/google/sentencepiece) 
### 方法介绍
SentencePiece包含以下4个主要部件：[`Normalizer`](#2)、[`Trainer`](#3)、[`Encoder`](#4) 和 [`Decoder`](#5)
#### Normalizer

#### Trainer

#### Encoder

#### Decoder


#### 示例代码
```python
--vocab_size=<size>             # 词典大小|V|, 在训练时设置
--normalization_rule_name=nfkc  # 正则化规则, 在训练时设置
--normalization_rule_tsv=<file> # hard convert map(最长匹配优先), str1 <tab> str2, 在训练时设置
```
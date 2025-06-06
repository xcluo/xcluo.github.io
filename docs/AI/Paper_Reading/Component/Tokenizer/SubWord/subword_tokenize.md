
### [ULM](https://arxiv.org/abs/1804.10959)
ULM 为概率模型驱动，将子词切分视为一个概率生成问题，通过统计语料库中子词的出现频率，建立 Unigram 语言模型，计算所有可能的子词切分概率，找到最可能的子词序列。
#### 基本原理
1. 初始化词表：从训练语料中统计所有字符和常见子串，作为初始候选词表。  
2. 训练Unigram模型：通过EM算法迭代优化子词概率 $p(w_i)$，**使语料的似然最大**

    - E-Step：对每个句子$x$，使用动态规划或Viterbi算法枚举所有可能的子词切分 $S(x)$，并计算各种切分$w$的概率

        $$
        \begin{aligned}
            P(w\vert x) =& \frac{\prod_{i=1}^{\vert w \vert} p(w_i)}{\sum_{w^{'}\in S(x)}\prod_{i=1}^{\vert w^{'} \vert} p(w^{'}_i)} \\
            c(w_i) =& \sum_{x\in X} \sum_{w \in S(x)} P(w\vert x)\cdot \text{count}(w_i, x)
        \end{aligned}
        $$

        > $p(w_i)$ 为当前子词 $w_i$ 的概率（初始化为均匀分布或频率统计）  
        > $\text{count}(w_i, x)$ 为子词 $w_i$ 在句子 $x$ 切分中的出现次数

    - M-Step，更新子词概率$p(w_i)$，最大化语料似然

        $$
        p(w_i) = \frac{c(w_i)}{\sum_{w^{'}_i \in V} c(w^{'}_i)}
        $$

3. 交替执行步骤E-Step和M-Step，直到子词概率收敛或达到指定迭代次数
4. 剪枝词表：保留概率最高的Top-K 子词（如8K~32K）

!!! info 
    分词时对应分数最高的 $p(w\vert x)$ 即为分词结果

#### Subword Regularization
Subword Regularization 是 ULM 的扩展技术，基于训练好的ULM分词器，**通过引入随机子词切分**，使模型学会对不同切分生成一致的表征，提升分词模型的鲁棒性和泛化能力。核心思想如下：

1. 采样候选的多分词，从句子$N$个可能的子词切分按^${1/\alpha}$归一化后的概率采样，而非固定选择分数最高切分。

    $$
    P_\text{sample}(w\vert x) \propto P(w\vert x) ^{1/\alpha}
    $$

2. 动态噪声注入，在每个 epoch（或 batch）为同一句子选择不同的切分输入语言模型，使模型学会对不同切分生成一致的表征。

!!! info 
    本质上在tokenizer过程中进行了数据增强，即将同一文本划分为不同的subword序列


### WordPiece
WordPiece需要前缀`##`作为中间subword标志，因此需要预先分词。
#### 基本原理
算法流程如下：

1. 初始化：对训练语料进行分割为单词（如按空格、标点符号等分割）再进一步拆分单词为subword序列  
2. 统计所有可能的subword-pair 的共现频次，选择似然分数$score$最大的subword-pair合并为新的subword

    $$
    score = \frac{\text{freq}(\text{subword-pair})}{\big(\text{freq}(\text{pair-left}) + \text{freq}(\text{pair-right})\big)}
    $$

3. 重复第2步直到subwords数达到最大$\vert V \vert$或当前step最高频的subword-pair频率为1，退出循环


#### 回退优化
=== "标准版"

    ```python
    for token in whitespace_tokenize(text):
        chars = list(token)
        if len(chars) > self.max_input_chars_per_word:
            output_tokens.append(self.unk_token)
            continue

        is_bad = False
        start = 0
        sub_tokens = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    cur_substr = substr
                    break
                # 贪心匹配时，未全匹配时回退1，直至回退至字符级
                end -= 1
            if cur_substr is None:
                is_bad = True
                break
            sub_tokens.append(cur_substr)
            # 相应回退
            start = end

        if is_bad:
            output_tokens.append(self.unk_token)
        else:
            output_tokens.extend(sub_tokens)
    ```

=== "优化版"

    优化版进一步处理局部oov导致整体[UNK]的毒性扩散情况，即  

    - 标准版：`ğxcluo → [UNK]`  
    - 优化版：`ğxcluo → [UNK] ##x ##c ##luo`

    ```python
    for token in whitespace_tokenize(text):
      chars = list(token)
      if len(chars) > self.max_input_chars_per_word:
        output_tokens.append(self.unk_token)
        continue

      # is_bad = False
      start = 0
      sub_tokens = []
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
          substr = "".join(chars[start:end])
          if start > 0:
            substr = "##" + substr
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1

        if cur_substr is None:
            # unify multiple-unk_token or one_unk-to-one_token
            if len(sub_tokens) == 0 or sub_tokens[-1] != self.unk_token:
                sub_tokens.append(self.unk_token)
            start += 1
        else:
            sub_tokens.append(cur_substr)
            start = end

      output_tokens.extend(sub_tokens)
    ```

### BPE
无需前缀`##`作为中间subword标记

#### 基本原理
算法流程如下：

1. 初始化：将所有文本序列**以字节Byte或字级别为单位**拆分，并在尾部添加一个停止符`</w>`
2. 统计各相邻 subword-pair 的频次，选择最高频次 subword-pair 合并成新的subword（subword-pair更新入合并规则），基于新subword更新subword-pair 的统计结果
3. 重复第2步直到subwords数达到最大$\vert V \vert$或当前step最高频的subword-pair频率为1，退出循环

```python
# 初始化：按照字节为单位划分文本序列，并在尾部添加停止符 </w>
{'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
# 基于subword-pair 共现频次对 subwords进行合并
  # 1. 最高频的subword-pair是`es`，共现 6+3=9 次，合并
    {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w es t </w>': 6, 'w i d es t </w>': 3}
  # 2. 最高频的subword-pair是`est`，共现 6+3=9 次，合并
    {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w est </w>': 6, 'w i d est </w>': 3}
  # 3. 最高频的subword-pair是`est</w>`，共现 6+3=9 次，合并
    {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w est</w>': 6, 'w i d est</w>': 3}
  # 4. 最高频的subword_pair是`lo`，共现 5+2=7 次，合并
    {'lo w </w>': 5, 'lo w e r </w>': 2, 'n e w est</w>': 6, 'w i d est</w>': 3}
  # ...
# 重复迭代直到subwords数达到词表大小|V|或当前step最高频的字节对频率为1，退出循环
```

#### BPE-Dropout
BPE-Dropout 在 BPE 的合并过程中引入随机性，以概率 $p$ 跳过某些合并步骤，从而生成同一单词的多种分词结果。

1. 合并规则：`"l" + "o" → "lo", "lo" + "w" → "low", "e" + "r" → "er"`
2. 基于合并规则，应用BPE-Dropout后，"lower"的可能分词为：
    - `"low" + "er"`（未跳过任何合并）
    - `"lo" + "w" + "er"` （跳过 `"lo" + "w" → "low"`）
    - `"l" + "o" + "w" + "er"` （跳过所有合并）

!!! success "优势"
    1. 对拼写错误、（大小写或形态）变体、多语言混合文本更鲁棒（如`"l0wer", "lOwer", "l〇wer"`）

- [sentencepiece subword regularization and BPE-dropout](https://github.com/google/sentencepiece?tab=readme-ov-file#subword-regularization-and-bpe-dropout)
#### 预先分词选择
 - [x] 处理包括大量固定属于的专业领域文本（医学、法律等）时
 - [x] 训练面向特定下游任务的tokenizer
 - [ ] 训练通用语言模型，处理多语言混合文本，追求最大灵活性和覆盖率

#### 固定高频词、维护领域词典

```python
def preprocess_chinese(text):
    # 领域词典
    fixed_phrases = {"北京", "上海", "人工智能", "机器学习"}
    for phrase in fixed_phrases:
        # 在tokenize时会使用white_split，以确保固定分词被划分
        text = text.replace(phrase, " " + phrase + " ")
    # 分词处理（可选）
    text = ' '.join(jieba.cut(text))
```
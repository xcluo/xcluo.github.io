
### ULM

#### 基本原理

### WordPiece
WordPiece需要前缀`##`作为中间subword标记，因此需要预先分词。可以回退到
#### 基本原理
算法流程如下：

1. 初始化：类似于BPE，对训练语料进行与分词（如按空格分割）并拆分  
2. 统计所有可能的subword-pair 的共现频次，选择似然分数最大的subword-pair合并为新的subword，基于新subword更新subword-pair 的统计结果  
    
    $$
    score = \frac{\text{freq}(\text{subword-pair})}{\big(\text{freq}(\text{pair-left}) + \text{freq}(\text{pair-right})\big)}
    $$

3. 重复第2步直到subwords数达到最大$\vert V \vert$或当前step最高频的subword-pair频率为1，退出循环


#### 可回退至字符级
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
无需前缀`##`作为单词中间subword标记

#### 基本原理
算法流程如下：

1. 初始化：将所有文本序列**以字节Byte或字级别为单位**拆分，并在尾部添加一个停止符`</w>`
2. 统计各相邻 subword-pair 的频次，选择最高频次 subword-pair 合并成新的subword，基于新subword更新subword-pair 的统计结果
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
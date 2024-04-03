

### tokenization
[`FullTokenizer`](https://github.com/google-research/bert/blob/master/tokenization.py#L161C11-L161C11)插入`special_token`

1. variety_span
2. 特殊字符切换为已经替换的某个token作为`relay_token`（`relay_token`为单字符unicode且左右各增加一个空格以确保整体分词为`relay_token`，而不是`##relay_token`）
3. tokenize
4. tokenized tokens还原，即`relay_token` → map → `target_token`
  > 可不recover，直接用该token表示特殊语义
!!! info 
    - 每种功能的`special_token`尽可能独立，比如`[SPACE]` 不和 `[PAD]`、`[SEP]`共用


### FullTokenizer优化

#### 去除声调上下标
```python title="BasicTokenizer._run_strip_accents"
# 去除字符声调 "ā á ǎ à"   -> "a a a a"
def _run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)
```

#### 避免oov毒性扩散
```python title="WordpieceTokenizer.tokenize"
# 优化：避免BPE分词oov毒性扩散现象，e.g., "玫瑰花𝖟lᴤ朵向日葵3Ⰻ7朵"
## [玫，瑰，花，unk，朵，向，日，葵，unk，朵] -> 
## [玫，瑰，花，unk, 1, unk，朵，向，日，葵，3, unk, ##7，朵]
def tokenize(self, text):
    text = convert_to_unicode(text)

    output_tokens = []
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
        # if cur_substr is None:
        #   is_bad = True
        #   break
        # sub_tokens.append(cur_substr)
        # start = end

        if cur_substr is None:
            if len(sub_tokens) == 0 \
              or sub_tokens[-1] != self.unk_token:  # unify multiple-unk_token or one_unk-to-one_token
                sub_tokens.append(self.unk_token)
            start += 1
        else:
            sub_tokens.append(cur_substr)
            start = end

      # if is_bad:
      #   output_tokens.append(self.unk_token)
      # else:
      #   output_tokens.extend(sub_tokens)

      output_tokens.extend(sub_tokens)
    return output_tokens
```
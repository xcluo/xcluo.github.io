

#### tokenization
[`FullTokenizer`](https://github.com/google-research/bert/blob/master/tokenization.py#L161C11-L161C11)插入special token

1. variety_span
2. 特殊字符切换为已经替换的某个`" variety_token "`（`variety_token`为单字符unicode且左右各增加一个空格以确保整体分词为一个token）
3. tokenize
4. tokenized tokens还原，即`variety_token` — map → `target_token`
### SubWord算法
SubWord算法，顾名思义就是把一个文本序列划分为更小的一个个**子词**（可能为单个字词的一部分，也可为长词）序列，即SubWords (或SubTokens)。

#### SubWord分词器
1. [WordPiece](SubWord/subword_tokenize.md#wordpiece)
2. [UnigramUnigram Language Model (ULM)](SubWord/subword_tokenize.md#ulm)，出自论文Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates
3. [BPE (Byte Pair Encoding)](SubWord/subword_tokenize.md#bpe)

- B；常见分词库tiktoken、sentencepiece
- [ ] pre_tokenizer：split分割句子，bytelevel拆分句子至字节级，因此BPE字典中不会有中文，反而有很多分解出的字节串，
- [ ] 很多BPE分词器会保留空格，因此需要预先将空格" "替换为一个不常用的字符，如replace(" ", "Ġ")，一般可以替换为一个词频少的char

#### 常用分词库
1. [SentencePiece](SubWord/SentencePiece.md)
2. [Tiktoken]：使用Rust语言编写，高度优化，速度比基于Python的标准BPE快3-6倍，前者直接使用预定义的编码词表，后者可以自行构建词表

todo：Pinyin Tokenizer、拆字、繁简、字素(字组成结构)、OCR
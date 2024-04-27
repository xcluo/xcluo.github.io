### 代码轮子
#### `data_utils`
```python title="data_utils.py"
# coding=utf-8
import tensorflow as tf
import json
from data_example import ExampleBuilder

class DataResourece:
    def __init__(self, 
                input_file, 
                shuffle, 
                epochs,
                drop_reainder,
                tokenizer_type,
                vocab_file,
                replace_file,
                max_seq_length,
                do_lower_case, 
                t2s,
                uni_white,
                truncate_direct):

        self.input_file = input_file
        self.shuffle = shuffle
        self.epochs = epochs
        self.drop_reainder = drop_reainder
        self.line_num = _count()

        self.builder = ExampleBuilder(tokenizer_type, vocab_file, 
                                      replace_file, max_seq_length, 
                                      do_lower_case, t2s, uni_white, 
                                      truncate_direct)

    # 由于使用生成器表达式，单次遍历统计样本数
    def _count(self):
        f = open(self.input_file, "r", encoding="utf-8")
        line_num = 0
        for line in f:
            try:
                json.loads(line)
            except json.decoder.JSONDecodeError:
                print(line)
                raise ValueError(f'line {line_num} is not a json sample')
            line_num += 1
        f.close()
        return line_num
    

    def generator(self):
        f = open(self.input_file, "r", encoding="utf-8")
        for line in f:
            try:
                line = json.loads(line)
            except json.decoder.JSONDecodeError:
                raise ValueError(f'line {i} is not a json sample')
            if "Content" in line.keys():
                content = line["Content"]
            elif "content" in line.keys():
                content = line["content"]
            elif "c" in line.keys():
                content = line["c"]
            else:
                raise ValueError

            if "Label" in line.keys():
                label = line["Label"]
            elif "label" in line.keys():
                label = line["label"]
            else:
                label = ""
            
            example = self.builder.build_example(text_a=content, text_b=None, 
                                                 text_label=label)

            # 对应dataset的`output_types`
            # todo: 软标签蒸馏时需要`str(label)`
            yield [content], [label], example.features["input_ids"], 
                  example.features["input_mask"], example.features["segment_ids"],
                  example.features["label_ids"]
    
    def next_batch(self, batch_size):
        dataset = tf.data.Dataset.from_generator(
            # 数据生成函数
            self.generator, 
            # 对应generator返回的数据类型
            output_types=(tf.string, tf.string, 
                          f.int32, tf.int32, tf.int32, tf.int32)
            )
        # repeat对应epochs，因此可基于全局
        dataset = dataset.repeat(self.epochs)
        # 训练时扰动、测试时顺序取值，设置扰动池大小
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=100 * batch_size)
        # 设置随机取值池大小
        dataset = dataset.prefetch(buffer_size=100 * batch_size)
        # 每次取值 `batch_size` 样本
        # `None`` 表示 `pad_to_longest`
        dataset = dataset.padded_batch(batch_size, padded_shapes=([None], [None], [None], [None], [None], [None], 
                                       drop_remainder=welf.drop_remainder))

        iterator = dataset.make_initializable_iterator()
        return iterator.get_next(), iterator.initializer
```
!!! info 
    - `padded_batch` 中如果采用了 `pad_to_longest` 对于处于不同 `batch_size` 中的同一样本，`longest_length` 的不同会导致 ==结果不会一致== ，但最终造成的区别影响可以忽略不记，<span style="color:green;">且不同数目的[PAD]能够更好地使模型学习到这个符号的语义和作用</span>。

#### `data_example`
```python title="data_example.py"
# coding=utf-8
import collections
import tokenization         # 自定义的一系列tokenization
import ahocorasick
from opencc import OpenCC
import re

class Example(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        input_len = len(input_ids)
        if not (input_len == len(input_mask) and input_len == len(segment_ids)):
            raise ValueError('All feature lists should have the same length ({})'.format(input_len))

        self.features = collections.OrderedDict([("input_ids", input_ids),
                                                 ("input_mask", input_mask),
                                                 ("segment_ids", segment_ids),
                                                 ("label_ids", label_ids)])


class ExampleBuilder(object):
    def __init__(self,
                 tokenizer_type,
                 vocab_file,
                 replace_file,
                 max_seq_length,
                 do_lower_case,
                 t2s=False,
                 uni_white=False,
                 truncate_direct='first'):

        if tokenizer_type == 'BPE':           # using BPE tokenizer
            self._tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=do_lower_case)
        elif tokenizer_type == 'char-level':  # using char-level tokenizer
            self._tokenizer = tokenization.CharTokenizer(vocab_file, do_lower_case=do_lower_case)
        # todo: 新增tokenizer
        else:
            raise ValueError(f'{tokenizer_type} is not a specialized tokenizer')

        self._pad_id = self._get_pad_id()

        self.trie = ahocorasick.Trie(allow_overlaps=False)  # AC算法执行字段替换
        self.replace_map = {}
        self._load_replace_file(replace_file)

        self.t2s = t2s
        self.cc = OpenCC("t2s")  # 中文繁简体转换
        
        self._max_seq_length = max_seq_length
        self.uni_white = uni_white
        self.truncate_direct = truncate_direct

        # todo: 新增空白字符
        self.white_space_pattern = re.compile(
            "[\u0000-\u001f\u007f\ufff0-\ufff8\\s\u3000\u00a0\u2002-\u200a\u202f\u205f]+")

        # todo: 手动设置 `special_tokens`
        self.special_tokens = [
            'unified_emojis',
            # "unified_white_chars"
        ]
        # todo: 为被tokenizer完整识别，设置`special toekn`的中转token
        self.special_token_map = {
            token: chr(ord('①') + i) for i, token in enumerate(self.special_tokens)
        }
        # todo: 设置`specia_token`的最终token
        self.special_token_result = {self.special_token_map[token]: f'[unused{i}]' for i, token in enumerate(self.special_tokens, 1)}

    def _truncate_first_list(self, x):
        return x[:self._max_seq_length - 2]

    def _truncate_last_list(self, x):
        begin = max(0, len(x) - self._max_seq_length + 2)
        return x[begin:]

    def _truncate_seq_pair(self, tokens_a, tokens_b):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= (self._max_seq_length - 3):
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _get_pad_id(self):
        try:
            return self._tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        except KeyError:
            return 0

    def _load_replace_file(self, replace_file):
        if replace_file is None:
            print("don't use replace file.")
            return
        with open(replace_file, "r", encoding="utf-8") as f:
            for line in f:

                line = line.strip()
                if len(line) == 0:
                    continue
                line_split = line.split("\t", maxsplit=1)
                assert len(line_split) == 2
                if line_split[0] in self.replace_map:
                    print('---' * 10, f'duplicate: {line_split[0]}', '---' * 10)
                self.replace_map[line_split[0]] = line_split[1]

            self.trie.build_trie(self.replace_map.keys())
        print("load %d replace token" % len(self.replace_map))

    # todo: 为完整保留原始特征，未提前执行 `str.lower()`, replace_file须同时包含大小写格式
    def replace_span(self, text):
        intervals = self.trie.parse_text(text)
        text_new = ""
        idx = 0
        for interval in intervals:
            text_new += text[idx:interval.start]
            token = text[interval.start:interval.end + 1]
            replace_token = self.replace_map[token]
            text_new += replace_token
            idx = interval.end + 1
        text_new += text[idx:]
        return text_new

    def uni_white_space(self, text):
        return re.sub(self.white_space_pattern, ' ', text.strip())

    # tokenize前中转特殊空白字符 && tokenize后映射特殊字符
    def preprocess(self, text):
        text = self.replace_span(text)
        if self.t2s:
            text = self.cc.convert(text)
        if self.uni_white:
            text = self.uni_white_space(text)
            if 'unified_white_chars' in self.special_tokens:
                text = re.sub(' ', self.special_token_map.get("unified_white_chars", ' '), text)
        # 避免 ##special_token 现象
        text = self.encode_special_tokens(text)
        return text

    def encode_special_tokens(self, text):        
        for key in self.special_token_map:
            text = text.replace(self.special_token_map[key], f' {self.special_token_map[key]} ')
        return text

    def decode_special_tokens(self, tokens):
        final_tokens = []
        for token in tokens:
            final_tokens.append(self.special_token_result.get(token, token))
        return final_tokens

    def build_example(self, text_a, text_b=None, text_label=None):
        text_a = self.preprocess(text_a)
        tokens_a = self._tokenizer.tokenize(text_a)

        tokens_b = None
        if text_b:
            text_b = self.preprocess(text_b)
            tokens_b = self._tokenizer.tokenize(text_b)
            tokens_b = self.decode_special_tokens(tokens_b)

        if self._max_seq_length:
            if tokens_b:
                self._truncate_seq_pair(tokens_a, tokens_b)
            else:
                if self.truncate_direct == 'first':
                    tokens_a = self._truncate_list(tokens_a)
                elif self.truncate_direct == 'last':
                    tokens_a = self._truncate_last_list(tokens_a)
                else:
                    raise ValueError(f'{self.truncate_direct} is not a specialized truncate direction')
            tokens_a = self.decode_special_tokens(tokens_a)

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # padding
        for _ in range(len(tokens), self._max_seq_length):
            input_ids.append(self._get_pad_id())
            input_mask.append(0)
            segment_ids.append(0)

        # todo: customize label definition
        if text_label:
            text_label_split = text_label.split(",")
            label_ids = [0]
            if "advertise" in text_label_split:
                label_ids[0] = 1
        else:
            label_ids = [0]

        example = Example(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids)
        return example
```
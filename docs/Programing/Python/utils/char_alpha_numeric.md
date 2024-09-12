```python
import unicodedata
import codecs
import string
import numpy as np
import json
from zhon import hanzi
import pypinyin
import re
from ahocorasick import Trie

## StringUtils ##
"""
    # staticmethod && classmethod #
        - parse_escape_text: 将未转义的字符串转化为转义后的结果
        - get_max_repeat_element: 获取字符串中最长连续字符长度
        - has_subsequent_spans: span_list中的字符串是否顺序存在于text中
        - has_all_spans
        - get_continue_arabic: 返回数字序列中连续的序列
        - get_continue_alpha: 返回字母序列中连续的单字母序列
        - get_continues: 返回连续的正、逆序数字和正、逆序单字母
        - slice_text: 对长文本text进行切片处理
        - is_chinese
        - is_japanese
        - is_korean


    # class && private_class
        - SpanReplacement
"""


class StringUtils:
    @staticmethod
    def parse_escape_text(text):
        return codecs.escape_decode(text)[0].decode("utf-8")

    @staticmethod
    def get_max_repeat_element_num(text, do_lower_case=True):
        if do_lower_case:
            text = text.lower()
        if len(text) == 0:
            return 0
        num = 1
        max_num = 1
        for i in range(1, len(text)):
            if text[i - 1] != text[i]:
                max_num = max(max_num, num)
                num = 1
            else:
                num += 1
        return max(max_num, num)

    @staticmethod
    def has_subsequent_spans(span_list, text):
        if not isinstance(span_list, (tuple, list)):
            raise ValueError(f"span_list is not tuple or list, but {type(span_list)}")
        if len(span_list) == 0 or len(text) == 0:
            return False
        cur_idx = 0
        for span in span_list:
            g = re.search(span, text)
            if not g:
                return False
            cur_idx += g.end()
            text = text[g.end():]
        return True

    @staticmethod
    def has_all_spans(span_list, text):
        if not isinstance(span_list, (tuple, list, set)):
            raise ValueError(f"span_list is not tuple or list or set, but {type(span_list)}")
        for span in set(span_list):
            if not re.search(span, text):
                return False
        return True
    
    @staticmethod
    def has_continues(text, n=3, step=1):
        ret1 = StringUtils.get_continue_arabic(text, n, step)
        reverse_ret1 = StringUtils.get_continue_arabic(text[::-1], n, step)
        ret2 = StringUtils.get_continue_alpha(text, n)
        reverse_ret2 = StringUtils.get_continue_alpha(text[::-1], n)
        return ret1 + reverse_ret1 + ret2 + reverse_ret2

    @staticmethod
    def get_continue_arabic(text, n=3, step=1):
        arabic_spans = [int(v) for v in re.findall('[0-9]+', text)]
        if not arabic_spans:
            return []

        sequences = [[arabic_spans[0]]]
        for span in arabic_spans[1:]:
            en_queue = False
            for seq in sequences:
                if seq[-1] + step == span:
                    seq.append(span)
                    en_queue = True
                    break
            if not en_queue:
                sequences.append([span])

        ret = [seq for seq in sequences if len(seq) >= n]
        return ret

    @staticmethod
    def get_continue_alpha(text, n=3):
        alpha_chars = [v for v in re.findall('[a-z]+', text) if len(v) == 1]
        if not alpha_chars:
            return []

        sequences = [[alpha_chars[0]]]
        for char in alpha_chars[1:]:
            en_queue = False
            for seq in sequences:
                if ord(seq[-1]) + 1 == ord(char):
                    seq.append(char)
                    en_queue = True
                    break
            if not en_queue:
                sequences.append([char])

        ret = [seq for seq in sequences if len(seq) >= n]
        return ret

    @staticmethod
    def slice_text(text, slice_len=128, seps={'\n', '！', '？', '。', '?', '!', '.'}):
        # 可层级递归切片，先 (N*slice_len) 分段、再 (1*slice_len) 分句
        pieces = set()
        pre_idx = 0         # left_idx

        while pre_idx < len(text):
            # rfind返回text中的真实idx
            sep_idx = max([text.rfind(sep, pre_idx, pre_idx + slice_len) for sep in seps])

            if sep_idx == -1 or pre_idx + slice_len >= len(text):
                part = text[pre_idx: pre_idx+slice_len]
                pre_idx += slice_len
            else:
                part = text[pre_idx: sep_idx+1]
                pre_idx = sep_idx + 1

            pieces.add(part)
        return pieces

    @staticmethod
    def is_chinese(cp):
        if (cp >= 0x4E00 and cp <= 0x9FFF) \
            or (cp >= 0x3400 and cp <= 0x4DBF) \
            or (cp >= 0x20000 and cp <= 0x2A6DF) \
            or (cp >= 0x2A700 and cp <= 0x2B73F) \
            or (cp >= 0x2B740 and cp <= 0x2B81F) \
            or (cp >= 0x2B820 and cp <= 0x2CEAF) \
            or (cp >= 0xF900 and cp <= 0xFAFF) \
            or (cp >= 0x2F800 and cp <= 0x2FA1F):
            return True
        return False

    @staticmethod
    def is_japanese(cp):
        if (0x3000 <= cp and cp <= 0x303f
            or 0x3040 <= cp and cp <= 0x309f
            or 0x30a0 <= cp and cp <= 0x30ff):
            return True
        return False

    @staticmethod
    def is_korean(cp):
        if (0x1100 <= cp and cp <= 0x11ff
            or 0x3130 <= cp and cp <= 0x318f
            or 0xac00 <= cp and cp <= 0xd7af
            or 0xa960 <= cp and cp <= 0xa97f
            or 0xd7b0 <= cp and cp <= 0xd7ff):
            return True
        return False


    class SpanReplacement:
        def __init__(self, replace_span_file, allow_overlaps=False, case_insensitive=False):
            self.trie = Trie(allow_overlaps=allow_overlaps, case_insensitive=case_insensitive)
            self.replace_map = dict()
            self.replace_span_file = replace_span_file
            self.load_replace_file()

        def load_replace_file(self):
            span_set = set()
            with open(self.replace_span_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    line_split = line.split('\t', maxsplit=1)
                    assert len(line_split) == 2
                    if line_split[0] in span_set:
                        print('---' * 10, 'duplicate_span', '---' * 10)
                        print(line)
                    span_set.add(line_split[0])
                    self.replace_map[line_split[0]] = line_split[1]
                self.trie.build_trie(self.replace_map)
            print(f"SpanReplacement loads {len(self.replace_map)} replace tokens")

        def replace_span(self, text):
            intervals = self.trie.parse_text(text)
            text_new = []
            idx = 0
            for interval in intervals:
                text_new += text[idx: interval.start]
                span = text[interval.start: interval.end + 1]
                replace_span = self.replace_map[span]
                text_new.append(replace_span)
                idx = interval.end + 1
            text_new.append(text[idx:])
            return "".join(text_new)

        def add_keywords(self, keywords):
            self.trie.build_trie(keywords)


## AlphaUtils ##
"""
    # staticmethod && classmethod #
        - full_to_half
        - half_to_full
        - count_full_width_chars
        - is_half_alpha
        - is_full_alpha
        - is_alpha
"""


class AlphaUtils:
    @staticmethod
    def full_to_half(s):
        return unicodedata.normalize("NFKC", s)

    @staticmethod
    def half_to_full(s):
        n = []
        for char in s:
            num = ord(char)
            if 0x21 <= num <= 0x7e:
                num += 0xfee0
            num = chr(num)
            n.append(num)
        return ''.join(n)

    @staticmethod
    def count_full_width_chars(s):
        return sum([AlphaUtils.full_to_half(c) != c for c in s])

    @staticmethod
    def is_half_alpha(c):
        if len(c) != 1:
            return False
        elif ord('a') <= ord(c) <= ord('z') or ord('A') <= ord(c) <= ord('Z'):
            return True
        return False

    @staticmethod
    def is_full_alpha(c):
        if len(c) != 1:
            return False
        if ord('ａ') <= ord(c) <= ord('ｚ') or ord('Ａ') <= ord(c) <= ord('Ｚ'):
            return True
        return False

    @staticmethod
    def is_alpha(c):
        return AlphaUtils.is_half_alpha(c) or AlphaUtils.is_full_alpha(c)


## PyTokenizer ##
"""
    # staticmethod && classmethod #
        - pinyin
        - has_pinyin
        - has_subsequent_pinyins: pinyin_list中的pinyin子串是否顺序存在于text中
        - has_all_pinyins
        - split_un_pinyin: 将无拼音的字符串划分（以##为前缀与单字母拼音区分）
        - strip_pinyin_tone
        - arabic_to_pinyin

    # class && private_class
        - __init__: 使用自定义的pinyin文件进行pinyin转换
"""


class PyTokenizer:
    number_pinyin_map = {
        "0": "ling2", "０": "ling2",
        "1": "yi1", "１": "yi1",
        "2": "er4", "２": "er4",
        "3": "san1", "３": "san1",
        "4": "si4", "４": "si4",
        "5": "wu3", "５": "wu3",
        "6": "liu4", "６": "liu4",
        "7": "qi1", "７": "qi1",
        "8": "ba1", "８": "ba1",
        "9": "jiu3", "９": "jiu3",

    }

    @staticmethod
    def pinyin(
            text,
            remain_alpha=False,
            remain_arabic=False,
            arabic_to_pinyin=False,
            white_list_chars={},
            py_tokenizer=None
    ):
        # split un-pinyin span
        if py_tokenizer == None:
            pinyins = pypinyin.lazy_pinyin(text, errors=lambda x: PyTokenizer.split_un_pinyin(x))
        else:
            pinyins = py_tokenizer.lazy_pinyin(text)
        new_pinyin = []
        for py in pinyins:
            if py.startswith('##'):
                if remain_alpha and AlphaUtils.is_alpha(py[-1]) \
                        or remain_arabic and NumericUtils.is_arabic(py[-1]) \
                        or py[-1] in white_list_chars:
                    py = py[-1]
                else:
                    continue
            new_pinyin.append(py)
        if remain_arabic and arabic_to_pinyin:
            new_pinyin = PyTokenizer.arabic_to_pinyin(new_pinyin)
        return new_pinyin

    @staticmethod
    def has_pinyin(
            text,
            pattern,
            match_type="fuzzy",  # strict or fuzzy
            arabic_to_pinyin=False,
            remain_alpha=False,
            remain_arabic=False,
            white_list_chars={},
            py_tokenizer=None):
        text_pinyin = PyTokenizer.pinyin(text, remain_alpha=remain_alpha, remain_arabic=remain_arabic,
                                         arabic_to_pinyin=arabic_to_pinyin,
                                         white_list_chars=white_list_chars, py_tokenizer=py_tokenizer)

        if match_type == "strict":
            text_pinyin = " ".join(text_pinyin)
        elif match_type == "fuzzy":
            text_pinyin = "".join(text_pinyin)
        else:
            raise ValueError("match_type must in {strict, fuzzy}")
        return re.findall(pattern, text_pinyin, re.I)

    @staticmethod
    def has_subsequent_pinyins(
            text,
            pinyin_list,
            match_type="fuzzy",  # strict or fuzzy
            arabic_to_pinyin=False,
            remain_alpha=False,
            remain_arabic=False,
            white_list_chars={},
            py_tokenizer=None
    ):
        if not isinstance(pinyin_list, (tuple, list)):
            raise ValueError(f"pinyin_list is not tuple or list, but {type(pinyin_list)}")
        if len(pinyin_list) == 0:
            return False
        text_pinyin = PyTokenizer.pinyin(text, remain_alpha=remain_alpha, remain_arabic=remain_arabic,
                                         arabic_to_pinyin=arabic_to_pinyin,
                                         white_list_chars=white_list_chars, py_tokenizer=py_tokenizer)
        if len(text_pinyin) == 0:
            return False
        if match_type not in {"strict", "fuzzy"}:
            raise ValueError("match_type must in {strict, fuzzy}")
        elif match_type == "strict":
            cur_idx = 0
            text_pinyin = " ".join(text_pinyin)
            for py_span in pinyin_list:
                g = re.search(f'(?:^| ){py_span}(?:$| )', text_pinyin)
                if not g:
                    return False
                cur_idx += g.end() + 1
                text_pinyin = text_pinyin[g.end():]
            return True
        elif match_type == "fuzzy":
            text_pinyin = "".join(text_pinyin)
            return StringUtils.has_subsequent_spans(text_pinyin, pinyin_list)

    @staticmethod
    def has_all_pinyins(
            text,
            pinyin_list,
            match_type="fuzzy",  # strict or fuzzy
            arabic_to_pinyin=False,
            remain_alpha=False,
            remain_arabic=False,
            white_list_chars={},
            py_tokenizer=None
    ):
        if not isinstance(pinyin_list, (tuple, list, set)):
            raise ValueError(f"pinyin_list is not tuple or list or set, but {type(pinyin_list)}")
        if len(pinyin_list) == 0:
            return False
        text_pinyin = PyTokenizer.pinyin(text, remain_alpha=remain_alpha, remain_arabic=remain_arabic,
                                         arabic_to_pinyin=arabic_to_pinyin,
                                         white_list_chars=white_list_chars, py_tokenizer=py_tokenizer)
        if len(text_pinyin) == 0:
            return False
        if match_type not in {"strict", "fuzzy"}:
            raise ValueError("match_type must in {strict, fuzzy}")
        elif match_type == "strict":
            text_pinyin = " ".join(text_pinyin)
            for py_span in pinyin_list:
                g = re.search(f'(?:^| ){py_span}(?:$| )', text_pinyin)
                if not g:
                    return False
            return True
        elif match_type == "fuzzy":
            text_pinyin = "".join(text_pinyin)
            return StringUtils.has_all_spans(pinyin_list, text_pinyin)

    @staticmethod
    def split_un_pinyin(text):
        new_pinyin = []
        for c in text:
            new_pinyin.append('##' + c)
        return new_pinyin

    @staticmethod
    def strip_pinyin_tone(py):
        if py.endswith('1') or py.endswith('2') or py.endswith('3') or py.endswith('4'):
            py = py[:-1]
        return py

    @classmethod
    def arabic_to_pinyin(cls, text, ignore_tone=True):
        new_pinyin = []
        for c in text:
            if c not in cls.number_pinyin_map:
                c = c
            else:
                c = cls.number_pinyin_map[c]
                if ignore_tone:
                    c = PyTokenizer.strip_pinyin_tone(c)
            new_pinyin.append(c)
        return new_pinyin

    def __init__(self,
                 pinyin_file,
                 ignore_tone=True,
                 file_type="JSON"  # JSON or CSV_key-idx_py_idx
                 ):
        self.ignore_tone = ignore_tone
        self.pinyin_map = self.load_pinyin_file(pinyin_file, file_type)

    def load_pinyin_file(self, pinyin_file, file_type):
        pinyin_map = dict()
        key_idx = -1
        py_idx = -1
        if file_type.lower().startswith("csv"):
            parts = file_type.split('_')
            key_idx = int(parts[1])
            py_idx = int(parts[-1])
        elif file_type.lower() == "json":
            pass
        else:
            raise ValueError(f"pinyin file type must be json or CSV_key-idx_py_idx")
        with open(pinyin_file, 'r', encoding='utf-8') as f:
            for line in f:
                if key_idx < 0:
                    line = json.loads(line)
                else:
                    parts = line.split()
                    key = parts[key_idx]
                    val = parts[py_idx].split(';')[0]
                    # skip non-pinyin token
                    if val == "null":
                        continue
                    if self.ignore_tone:
                        val = PyTokenizer.strip_pinyin_tone(val)
                    line = {key: val}
                for key, val in line.items():
                    if key in pinyin_map:
                        print(
                            f'{key} has already in current pinyin file\n\tprevious: {pinyin_map[key]}\n\tcurrent: {val}')
                    pinyin_map[key] = val
        print(f'pinyin file loads {len(pinyin_map)} samples')
        return pinyin_map

    def lazy_pinyin(self, text):
        text = text.lower()
        pinyins = []
        for c in text:
            if c in self.pinyin_map:
                c = self.pinyin_map.get(c)
            else:
                c = "##" + c
            pinyins.append(c)
        return pinyins


## NumericUtils ##
"""
    # staticmethod && classmethod #
        - uni_to_arabic
        - is_half_arabic
        - is_full_arabic
        - is_arabic
        - get_float_length: 返回浮点型变量小数点位数
"""


class NumericUtils:
    number_map = {
        "零": "0",
        "一": "1", "壹": "1",
        "二": "2", "贰": "2",
        "三": "3", "叁": "3",
        "四": "4", "肆": "4",
        "五": "5", "伍": "5",
        "六": "6", "陆": "6",
        "七": "7", "柒": "7",
        "八": "8", "捌": "8",
        "九": "9", "玖": "9",
        "拾": "十",
        "佰": "百",
        "仟": "千",
    }

    @classmethod
    def uni_to_arabic(cls, text, extra_number_map=dict(), extra_first=True):
        new_text = []
        for c in text:
            if extra_first:
                c = extra_number_map.get(c, cls.number_map.get(c, c))
            else:
                c = cls.number_map.get(c, extra_number_map.get(c, c))
            new_text.append(c)
        return "".join(new_text)

    @staticmethod
    def is_half_arabic(c):
        if len(c) != 1:
            return False
        elif ord('0') <= ord(c) <= ord('9'):
            return True
        return False

    @staticmethod
    def is_full_arabic(c):
        if len(c) != 1:
            return False
        elif ord('０') <= ord(c) <= ord('９'):
            return True
        return False

    @staticmethod
    def is_arabic(c):
        return NumericUtils.is_half_arabic(c) or NumericUtils.is_full_arabic(c)

    @staticmethod
    def get_float_length(val):
        val_str = np.format_float_positional(val, trim='-')
        n = len(val_str.split('.')[-1])
        return n


## PunctuationUtils ##
"""
    # staticmethod && classmethod #
        - strip_punctuation
        - strip_white_space
"""


class PunctuationUtils:
    all_punctuation = set(list(string.punctuation + hanzi.punctuation))
    all_punctuation.remove("\u3000")

    @classmethod
    def strip_punctuation(cls, text, white_list_punctuation={}):
        ret = []
        for c in text:
            if c not in white_list_punctuation \
                    and (c in cls.all_punctuation or
                         unicodedata.category(c).startswith("P")):
                continue
            ret.append(c)
        return "".join(ret)

    @staticmethod
    def strip_white_space(text, white_list_space={}, replace_token=""):
        ret = []
        for c in text:
            cp = ord(c)
            if c not in white_list_space and \
                    (
                            0x00 <= cp <= 0x20 or cp == 0x3000
                            or 0x7f <= cp <= 0xa0
                            or cp == 0x034f
                            or 0x2000 <= cp <= 0x200f or cp == 0x2011 or 0x2028 <= cp <= 0x202f or 0x205f <= cp <= 0x206f
                            or 0xfe00 <= cp <= 0xfe0f
                            or 0xe0100 <= cp <= 0xe01ef
                            or cp == 0xfeff
                            or cp == 0x115f or cp == 0x1160 or cp == 0x3164 or cp == 0xffa0
                            or 0xfff0 <= cp <= 0xffff
                            or 0xe0000 <= cp <= 0xe007f
                            or unicodedata.category(c) in ("Zs",)
                            # Cc/Cf
                    ):
                c = replace_token
            ret.append(c)
        return "".join(ret)
```
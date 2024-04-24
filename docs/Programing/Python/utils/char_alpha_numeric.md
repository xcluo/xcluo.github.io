```python
import unicodedata
import codecs
import string
from zhon import hanzi


## char_token_related ##
"""
    - parse_escape_text: 将未转义的字符串转化为转义后的结果
    - get_max_repeat_element: 获取字符串中最长连续字符长度
"""
class string_related:
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
        pre = text[0]
        max_num = 1
        for i in range(1, len(text)):
            if text[i-1] != text[i]:
                max_num = max(max_num, num)
                num = 1
            else:
                num += 1
        return max(max_num, num)



## alpha_related ##
"""
    - full_to_half
"""
class alpha_related:
    @staticmethod
    def full_to_half(s):
        return unicodedata.normalize("NFKC", s)


## numeric_related ##
"""
    - uni_numer_to_numeric
"""
class numeric_related:
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
        }

    @classmethod
    def uni_to_arabic(cls, text, extra_number_map=dict()):
        new_text = []
        for c in text:
            # c = cls.number_map.get(c, extra_number_map.get(c, c))
            c = extra_number_map.get(c, cls.number_map.get(c, c))
            new_text.append(c)
        return "".join(new_text)


## punctuation_related ##
"""
    - strip_punctuation
    - strip_white_space
"""
class punctuation_related:
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
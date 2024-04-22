import unicodedata


## char_token_related ##
"""
    - 
"""

## alpha_related ##
"""
    - full_to_half
"""

def full_to_half(s):
    return unicodedata.normalize("NFKC", s)


## numeric_related ##
"""
    - uni_numer_to_numeric
"""

def uni_number_to_numeric(text):
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
    new_text = []
    for c in text:
        c = number_map.get(c, c)
        new_text.append(c)
    return "".join(new_text)


## punctuation_related ##
"""
    - strip_punctuation
    - strip_white_space
"""

def strip_punctuation(text, white_list_punct={}):
    ret = []
    for c in text:
        cp = ord(c)
        if c not in white_list_punct and \
            (33 <= cp <= 47 or
            58 <= cp <= 64 or
            91 <= cp <= 96 or
            123 <= cp <= 126 or
            unicodedata.category(c).startswith("P")):
            continue
        ret.append(c)
    return "".join(ret)


def strip_white_space(text, white_list_white_space={}):
    ret = []
    for c in text:
        cp = ord(c)
        if c not in white_list_white_space and \
            (33 <= cp <= 47 or
            58 <= cp <= 64 or
            91 <= cp <= 96 or
            123 <= cp <= 126 or
            unicodedata.category(c).startswith("Z")):
            continue
        ret.append(c)
    return "".join(ret)
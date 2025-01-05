```python
from pypinyin import pinyin, Style

def pinyin(
    hans,                            # Union(List[str], str), 待拼音化的汉字字符串或字符串列表
    style=Style.TONE,                # 指定拼音风格, TONE表示声调在韵母上面
    heteronym=False,                 # 是否启用多音字, False只返回常用读音
    errors='default',                # Union(Callable, str)，无拼音字符处理方式, {default: 保留; ignore: 忽略; replace: 替换为}
    strict=False,                    # 是否根据上下文来选择最合适的拼音读法
    v_to_u=False,                    # 是否用v替代ü
    neutral_tone_with_five=False,    # 轻声是否用5表示
) -> List[List[Str]]:

def lazy_pinyin(
    hans,                            # Union(List[str], str), 待拼音化的汉字字符串或字符串列表
    style=Style.NORMAL,              # 指定拼音风格, NORMAL表示不带声调
    errors='default',                # Union(Callable, str)，无拼音字符处理方式, {default: 保留; ignore: 忽略; replace: 替换为}
    strict=False,                    # 是否根据上下文来选择最合适的拼音读法
    v_to_u=True,                     # 是否用v替代ü
    neutral_tone_with_five=False,    # 轻声是否用5表示
) -> List[List[Str]]:
```

#### has_pinyin
```python title="has_pinyin"
letter_offset_map = {'semi-lower': ord('a'), 'semi-upper': ord('A'), 'full-lower': ord('ａ'), 'full-upper': ord('Ａ')}
numeric_offset_map = {'semi-numeric': ord('0'), 'full-numeric': ord('０')}


def has_pinyin(
    text, 
    pattern,                # 拼音格式，支持正则匹配
    style=Style.NORMAL, 
    remain_letter=False,    # 是否保留文本中原有（大、小写，全半角）字母
    remain_numeric=False    # 是否保留文本中原有（全半角）数字
    ):
    text_pinyin = get_pinyin(text, style=style, remain_letter=remain_letter, remain_numeric=remain_numeric)
    if '|' in pattern or \
            '?' in pattern or \
            isinstance(pattern, str) and re.findall('^[a-z\?\)\(:]+$', pattern, re.I):
        pattern_pinyin = pattern
    else:
        pattern_pinyin = get_pinyin(pattern, style=style, remain_letter=remain_letter, remain_numeric=remain_numeric)

    return len(re.findall(pattern_pinyin, text_pinyin, re.I)) > 0


def get_pinyin(text, style=Style.NORMAL, remain_letter=False, remain_numeric=False):
    if remain_letter or remain_numeric:
        # {default: 保留原字符; ignore: 忽略并跳过}
        pinyin_list = pinyin(text, style=style, errors='default')

        if remain_letter:
            pinyin_list = [
                [chr(ord('a') + ord(py[0]) - letter_offset_map[is_letter(py[0])])] if is_letter(py[0]) else py for py in
                pinyin_list]
        else:
            pinyin_list = [py for py in pinyin_list if not is_letter(py[0])]

        if remain_numeric:
            pinyin_list = [
                [chr(ord('0') + ord(py[0]) - numeric_offset_map[is_numeric(py[0])])] if is_numeric(py[0]) else py for py
                in pinyin_list]
        else:
            pinyin_list = [py for py in pinyin_list if not is_numeric(py[0])]

    else:
        pinyin_list = pinyin(text, style=style, errors='ignore')

    return ''.join([py[0] for py in pinyin_list])


def is_letter(c):
    if len(c) != 1: # 保留的非拼音字符只可能为len=1的list
        return 0
    if ord('a') <= ord(c) <= ord('z'):
        return 'semi-lower'
    elif ord('A') <= ord(c) <= ord('Z'):
        return 'semi-upper'
    elif ord('ａ') <= ord(c) <= ord('ｚ'):
        return 'full-lower'
    elif ord('Ａ') <= ord(c) <= ord('Ｚ'):
        return 'full-upper'
    else:
        return 0


def is_numeric(c):
    if len(c) != 1: # 保留的非拼音字符只可能为len=1的list
        return 0
    if ord('0') <= ord(c) <= ord('9'):
        return 'semi-numeric'
    elif ord('０') <= ord(c) <= ord('９'):
        return 'full-numeric'
    else:
        return 0
```


#### count_part_elements
```python
# 每段首字母为数字谐音的个数
def count_part_elements(text):
    text_no_punct = re.sub(punctuation, ' ', text)
    text_uni_space = re.sub(white_space_pattern, ' ', text_no_punct.strip())
    parts = text_uni_space.strip().split()
    flags = [has_pinyin(part.lower(), '(?:lin|yi|yao|er|lia|shan|sa|sh?i|wu|li?u|qi|ba|bie|jiu)|[0-9０-９]', remain_numeric=True, remain_letter=True) for part in parts]
    return sum(flags)

# 每段首字母是否连续
def count_subsequent_part_elements(text):
    subsequent_pattern = '(?:yi|1|１)(?:er|liang|lia|2|２)(?:san?|sh?an|3|３)' \
                         '|(?:er|liang|2|２)(?:san?|sh?an|3|３)(?:sh?i|4|４)' \
                         '|(?:san?|sh?an|3|３)(?:sh?i|4|４)(?:wu|5|５)' \
                         '|(?:sh?i|4|４)(?:wu|5|５)(?:li?u|6|６)' \
                         '|(?:wu|5|５)(?:li?u|6|６)(?:qi|7|７)' \
                         '|(?:li?u|6|６)(?:qi|7|７)(?:ba|8|８)' \
                         '|(?:qi|7|７)(?:ba|8|８)(?:jiu|9|９)' \
                         '|(?:ba|8|８)(?:jiu|9|９)shi'
    # punct.split + white_space.split
    text_no_punct = re.sub(punctuation, ' ', text)
    text_no_space_punct = re.sub(white_space_pattern, ' ', text_no_punct.strip())
    parts = text_no_space_punct.strip().split()
    elements = ''.join([p[0] for p in parts])

    flag = has_pinyin(elements.lower(), subsequent_pattern, remain_numeric=True, remain_letter=True)
    if flag:
        return flag
    # white_space.split
    text_no_space_punct = re.sub(white_space_pattern, ' ', text.strip())
    parts = text_no_space_punct.strip().split()
    elements = ''.join([p[0] for p in parts])
    flag = has_pinyin(elements.lower(),
                      subsequent_pattern,
                      remain_numeric=True, remain_letter=True)
    return flag
```
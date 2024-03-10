```python
from pypinyin import pinyin, Style

def pinyin(
    hans,                            # 字符串或字符串列表
    style=Style.NORMAL,              # 指定拼音风格, 拼音格式，NORMAL表示不带声调
    v_to_u=False,                    # 是否用v替代ü
    neutral_tone_with_five=False,    # 轻声是否标注(用5表示)
    heteronym=False,                 # 是否启用多音字
    errors='default',                # 处理没有拼音oov字符方式, {default: 保留; ignore: 忽略; replace: 替换为}
    ) -> List[List[Str]]:


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
    letter_offset_map = {'semi-lower': ord('a'), 'semi-upper': ord('A'), 'full-lower': ord('ａ'), 'full-upper': ord('Ａ')}
    numeric_offset_map = {'semi-numeric': ord('0'), 'full-numeric': ord('０')}

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
```
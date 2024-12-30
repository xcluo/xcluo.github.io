```python
import emoji    # pip install emoji
```

#### 常用功能

1. `is_emoji`
判断输入的 `string` 是否为单个emoji

2. `emojize`
将带有左右冒号的英文表示转化为emoji
> `:thumbs_up: -> 👍`
3. `demojize`
将emoji转化为带有左右冒号的英文表示
> `👍 -> :thumbs_up:`
4. `emoji_list()` 返回list，包含字符串中所有emoji及其位置信息
   ```python
   for emj in emoji.emoji_list():
       print(emj['emoji'], emj['match_start'], emj['match_end'])   # [match_start, match_end)
   ```

!!! info ""
    - emoji存在颜色变种，即{本色, 微浅, 浅色, 中等, 微深, 深色}六种，可通过 `re.sub('_(?:dark|medium-dark|medium|medium-light|light)_skin_tone', '', de)` 进行归一化
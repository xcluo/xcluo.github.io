```python
# pip install googletrans==4.0.0-rc1
from googletrans import Translator
translator = Translator()
```

1. 翻译
```python
# Translated.text 为翻译后的文本
def translate(self, text, dest='en', src='auto') -> Translated:

# 获取翻译结果，`dest`指定翻译的目标语言
translator.translate(content, dest='zh-cn').text
```
2. 常用语种
```python
LANGUAGES = {
    'en': 'english',
    'zh-cn': 'chinese (simplified)',
    'zh-tw': 'chinese (traditional)',
    ...
}
```
3. 语种识别
```python
# Detected.lang 为检测的语种
def detect(self, text) -> Detected:
    pass
```
OpenCC：Open Chinese Convert

```python
from opencc import OpenCC       # pip install OpenCC

cc = OpenCC(config="t2s")       # t2s: 繁体转简体;      s2t: 简体转繁体;
                                # tw2s: 台湾正体转简体; s2tw: 简体转台湾正体;
                                # hk2s: 香港繁体转简体; s2hk: 简体转香港繁体;
                                # t2jp: 繁体转新日文;   jp2t: 新日文转繁体;

text = cc.convert(text)
```
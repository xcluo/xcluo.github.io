1. **反斜杠转义问题**
```python
import codecs
cnt = "仨竺汃昗宍竺訵宍泗\\n藤训号\\n"

cnt = codecs.escape_decode(cnt)[0].decode("utf-8")  # escape_decode返回(bytes, len(bytes))
                                                    # "仨竺汃昗宍竺訵宍泗\n藤训号\n"
```
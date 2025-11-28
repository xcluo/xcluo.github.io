jieba库分词的三种模式：
1. 精准模式`cut_all=False`：把文本精准地分开，不存在冗余
2. 全模式`cut_all=True`：把文中所有可能的词语都扫描出来，存在冗余
3. 搜索模式`lcut_for_search`：在精准模式的基础上，再次**对长词**进行切分，适用于搜索引擎分词

```python
import jieba
from jieba import analyse       # jieba.__init__中没有导入analyse包，因此需要单独导入

# --> 分词
# 缺省返回generator, 方法名 【l】开头表示转型为list
jieba.cut(sent, cut_all=False, HMM=False, use_paddle=False)     # 返回generator
jieba.lcut(sent, cut_all=False, HMM=False, use_paddle=False)    # 本质上在cut的基础上套用了list()
jieba.cut_for_search(sent, HMM=True)                            # 使用HMM进一步划分长词，返回generator
jieba.lcut_for_search(sent, HMM=True)                           # 本质上在cut_for_search的基础上套用了list()

# --> 外部字典
jieba.load_userdict(dict_path)              # 从文件中添加外部字典，每行csv，后二者可省略: 词\t词频\t磁性
jieba.add_word(word, freq=None, tag=None)   # 直接添加词

# --> 关键字抽取
analyse.
```
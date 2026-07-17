---
title: "Python"
---

#### 环境搭建

- [`conda`](tools/py_env.md#conda)
- [`uv`](tools/py_env.md#uv)
- [`poetry`](tools/py_env.md#poetry)

### 基础部分

1. [解释器命令行选项](basic/commond_line.md)
1. [`class`](basic/class.md) 类
1. [内置方法](basic/builtins.md)
1. [`typing`](basic/typing.md) 静态类型注解
1. [`Exception Handling`](basic/exception.md) 异常处理

### 细节部分

1. [`Container`容器类](details/container.md) 包括Iterable、Iterator和Generator
1. [`lambda`](details/lambda.md) 匿名函数
1. [`decorator`](details/decorator.md) 修饰符
1. [`fstring`](details/fstring.md), [`Jinja2`](details/jinja2.md) 字符串插值

### 自用库

#### 算法类

- [`Aho-Corasick`](libs/ahocorasick.md)：A-C算法，多模式匹配中的经典算法
- [`collections`](libs/collections.md)：Python内建的一个集合模块，提供了许多有用的集合类和方法。

#### 功能类

1. 并发调度相关  
    - 并发操作：[`threading`](libs/concurrent_programming.md#threading), [`multiprocessing`](libs/concurrent_programming.md#multiprocessing),  [`concurrent.futures`](libs/concurrent_programming.md#concurrentfeatures)
    - 异步操作：[`asyncio`](libs/async_operation.md#asyncio), [`aiofiles`](libs/async_operation.md#aiofiles), [`aiohttp`](libs/async_operation.md#aiohttp), [`async + await`]
    - [`schedule`](libs/schedule.md)：定时任务

- [`functools`](libs/functools.md)：提供一些高阶函数
- [`itertools`](libs/itertools.md)
- 文件、系统相关
    - [`os`](libs/os.md)：提供一些方便使用操作系统相关功能的函数
    - [`glob`](libs/os.md#glob)：文件匹配
- 终端控制  
    - [`sys`](libs/sys.md)：提供一些方便Python解释器
    - 代码传参：[`sys.argv`](libs/argparser.md#sysargv)，[`argparse`](libs/argparser.md#argparse)，[`tf.flags`](libs/argparser.md#tfflags)
    - 进度条显示：[`tqdm`](libs/tqdm.md)
    - [`colorama`](libs/colorama.md)控制终端输出
- [`pytest`](libs/pytest.md)：测试框架
- [`subprocess`](libs/subprocess.md)

#### 统计、绘图

- [`matplotlib`](libs/matplotlib.md)：提供数据绘图功能的第三方库
- [`wordlcoud`](libs/wordcloud.md)：绘制词汇组成类似云的彩色图形
  
#### 资源库

- [`emoji`](libs/emoji.md)：提供了一些emoji的相关操作，emoji收集新而全
- [`OpenCC`](libs/opencc.md)：包含中文繁简体转换库
- [`pypinyin`](libs/pypinyin.md)：汉字⟷拼音
- [`unicodedata`](libs/unicodedata.md)
- [`codecs`](libs/codecs.md)
- [`datetime`](libs/datetime.md)
- [`log`](libs/log.md)
- [`googletrans`](libs/googletrans.md)：google翻译api
- 图片相关：[`pdf2image`](libs/pdf2image.md)，[`rapidocr_onnxruntime, rapidocr, onnxocr`](libs/ocr_tools.md)

#### utils

- [`ahocorasick.py`](utils/ahocorasick.md)
- [`generate_regrex`](utils/generate_regrex.md)
- [`char_alpha_numeric.py`](utils/char_alpha_numeric.md)
- [`random_methods.py`](utils/random_methods.md)
- [`general_dataset_utils.py`](utils/general_dataset_utils.md), [`torch_dataset_utils.py`](utils/torch_dataset_utils.md), [`tf1_dataset_utils.py`](utils/tf1_dataset_utils.md)
- [`tokenization.py`](utils/tokenization.md)

### 进阶：文件读写

#### 数据格式

- [`bin`](libs/file_format.md#bin)：二进制文件访存
- [`json`](libs/file_format.md#json)：提供了在JSON数据和Python对象之间进行转换和序列化的功能。
- [`pickle`](libs/file_format.md#pkl)：Python专用自定义存储数据格式
- [`pandas`](libs/pandas.md)
- [`csv`](libs/xlsx.md#csv)：CSV文件读写
- [`xlsx`](libs/xlsx.md#xlsx)：excel文件读写
- [`pdf`](libs/pdf.md)

### 进阶：数据库操作

- DatabaseConnection
- mysql
- postgresql
- sqlite
- [milvus](database/milvus.md)向量数据库
- faiss向量数据库

### 进阶：AI Libraries

#### 机器学习

- [`sklearn`](ai_libs/sklearn/sklearn.md)
- [`gensim`](ai_libs/gensim/gensim.md)
- [`lightgbm`]()
- [`fasttext`] pip install fasttext-wheel

#### 分词方法

- BPE 分词器：[`sentencepiece`](ai_libs/bpe_tokenizer.md)、[`tokenization`](ai_libs/bpe_tokenizer.md)
- [`jieba`]()

### 进阶: 爬虫

- requests

#### 浏览器操作

- playwright

### 进阶：Web服务

- fastapi: ~3000请求/s
- flask: ~1000请求/s，同步WSGI (Web Server Gateway Interface)，更灵活更底层

### 进阶: GUI

1. [`tkinter`]()
1. [`pyqt`]()

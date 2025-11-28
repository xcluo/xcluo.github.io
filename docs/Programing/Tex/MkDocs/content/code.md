

#### Code Block
```
\```
sss
\```
```
https://squidfunk.github.io/mkdocs-material/reference/code-blocks/

- yaml
- python

```title="lxc"
print('hello world')
a = tf.zeros([3, 3], dtype=tf.float32)
```

#### Inline Code BLock
指定行内代码语言类型，`#!python import tensorflow as tf`
```
`#!python code_content`
```

#### [Embedding External Files](https://facelessuser.github.io/pymdown-extensions/extensions/snippets/#block-format)
需要在docs目录下新建snippets文件夹用于存放待插入的文件
```
- pymdownx.snippets:
    base_path: docs/snippets

# 多片段选择
-8<-
filename:start1:end1,start2:end2

filename:start1:end1,start2:end2
-8<- 
> 在filename前增加;意为注释掉相应的file插入

# 通过指定行号选择
-8<- filename:start1:end1,start2:end2

# 通过片段别名选择
-8<- filename:func

# --8<-- [start:func]
def my_function(var):
    pass
# --8<-- [end:func]
```







#### Code [Content Tabs](https://squidfunk.github.io/mkdocs-material/reference/content-tabs/)
将多个内容并列显示，常用于Code展示

=== "C"
    hello world
    ``` c
    #include <stdio.h>

    int main(void) {
      printf("Hello world!\n");
      return 0;
    }
    ```
=== "<span style="color: red">C++</span>"

    ``` c++
    #include <iostream>

    int main(void) {
      std::cout << "Hello world!" << std::endl;
      return 0;
    }
    ```

    ```python title="lxc"
    print("hello world")
    ```


#### 配置
```yaml
markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true       
      line_spans: __span
      pygments_lang_class: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.superfences

theme:
  features:
    - content.code.copy           # 代码块内容一键复制按键
    - content.code.annotate
```





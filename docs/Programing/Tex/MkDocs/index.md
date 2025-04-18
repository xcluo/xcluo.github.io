#### 主题


#### 部件
- [admonition](theme_related/admonition.md)


- [content tabs](https://squidfunk.github.io/mkdocs-material/reference/content-tabs/)
=== "C"
   
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

### xxx
https://squidfunk.github.io/mkdocs-material/reference/code-blocks/

指定行内代码语言类型
```
`#!python range()`
```

```title="lxc"
print('hello world')
a = tf.zeros([3, 3], dtype=tf.float32)
```

- ==This was marked==
- a^T^，a~T~
- +++
- ^^This was inserted^^
- ~~This was deleted~~
- {deleted}
- Text can be {--deleted--} and replacement text {++ added ++}. This can also be
combined into {~~one~>a single~~} operation. {==Highlighting==} is also
possible {>>and comments can be added inline<<}.


``` yaml
# (1)!
```

1.  Look ma, less line noise!

```
| 左对齐 | 右对齐 | 居中对齐 |
| :-----| ----: | :----: |
| 单元格 | 单元格 | 单元格 |
| 单元格 | 单元格 | 单元格 |
```


| 左对齐 | 右对齐 | 居中对齐 |
| :-----| ----: | :----: |
| 单元格 | 单元格 | 单元格 |
| 单元格 | 单元格 | 单元格 |


#### 相对路径
```python
# in mkdocs.yml
use_directory_urls: false

# link resource
"can use relative path and absolute path"

# link file
"for .md: add file_name suffix"
"for .md head: file_name.suffix#head_name"
```

#### 图片
```title="one-image"
<div class="one-image-container">
    <img src="image/FP32_demo.png" style="width: 80%;">
    <!-- <p>LoRA在Attention各部分权重上的消融实验效果</p> -->
    <!-- <figcaption>这是图片的标题或描述。</figcaption> -->
</div>
```

```title="row-image"
<div class="row-image-container">
    <div>
        <img src="image/FP32_demo.png" style="width: 80%;">
        <!-- <p>LoRA在Attention各部分权重上的消融实验效果</p> -->
        <!-- <figcaption>这是图片的标题或描述。</figcaption> -->
    </div>

    <div>
        <img src="image/FP32_demo.png" style="width: 80%;">
        <!-- <p>LoRA在Attention各部分权重上的消融实验效果</p> -->
        <!-- <figcaption>这是图片的标题或描述。</figcaption> -->
    </div>
</div>
```
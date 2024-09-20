
### 安装
```markdown
markdown_extensions:
  - admonition

# 设置不同类型的icons https://squidfunk.github.io/mkdocs-material/reference/admonitions/#admonition-icons-octicons
theme:
  icon:
    admonition:
      note: octicons/tag-16
      abstract: octicons/checklist-16
      info: octicons/info-16
      tip: octicons/squirrel-16
      success: octicons/check-16
      question: octicons/question-16
      warning: octicons/alert-16
      failure: octicons/x-circle-16
      danger: octicons/zap-16
      bug: octicons/bug-16
      example: octicons/beaker-16
      quote: octicons/quote-16
```

### [使用](https://squidfunk.github.io/mkdocs-material/reference/admonitions/#admonition-icons-octicons)
```markdown
# 不可折叠形式，title缺省为type_qualifier
!!! type_qualifier "qualifier_title"
    content
# 可折叠形式，缺省折叠
??? type_qualifier "qualifier_title"
    content
# 行内块，缺省行内左侧对齐，end控制右侧对齐
# 行内块需在正文内容之前定义
!!! type_qualifier inline end "qualifier_title"
    content

# html格式，可用来控制缩进提示
<div class="admonition note" style="margin-left: 20px;">
    <p class="admonition-title">qualifier_title</p>
    <p>content</p>
</div>  
```

!!! warning "lxc"
    hello world

    !!! warning "lxc"
        hello world


??? warning "lxc"
    hello world

hello world

!!! info inline "lxc"
    xxx



!!! info "$\sum\frac{\pi }{\sigma }-sin(x)$" 
    This is an admonition box without a title.

sss
sss  
xxx  
xxx
</br>
</br>


```html
<div class="admonition note" style="margin-left: 20px;">
    <p class="admonition-title">qualifier_title</p>
    <p>content</p>
</div>  
```



### [自定义警示类型](https://squidfunk.github.io/mkdocs-material/reference/admonitions/#customization)


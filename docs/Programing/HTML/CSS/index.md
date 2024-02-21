

#### 多级css

```
<!-- 层级关系：highlight.ol.li -->
.highlight ol li {
    ...
}
```

- .开头表示 class 选择器，如 `.class {...}`
- #开头表示 id 选择器，如 `#id {...}`
- 无开头表示 标签选择题，如 `p {...}`


#### 自定义标签
```
abbr {
  font-style: italic;
  color: chocolate;
}
<p>
  You can use <abbr>CSS</abbr> (Cascading Style Sheets) to style your <abbr>HTML</abbr> (HyperText Markup Language).
</p>
```
### giscus

#### 使用
1. `mkdocs.yml` 中配置，在blog目录所有post文件中激活
```yaml
theme: 
    custom_dir: overrides
```


2. 非post文件中激活
    ```yaml
    ---
    comments: true
    ---
    ```

#### 配置
1. Github开启讨论模块 `Settings -> General -> Features -> 勾选Discussion`
1. [install giscus](https://github.com/apps/giscus) 并使用Github授权
2. [configure giscus](https://giscus.app/zh-CN)

    - **语言**：简体中文
    - **enable giscus仓库**：`用户名/仓库名`
    - **页面↔️Discussion映射关系**：pathname
    - **Discussion 分类**：Announcements，确保只由仓库维护者和giscus创建
    - **特性**：全选
    - **启用giscus**：用于`comments.html`

3. 创建 `/overrides/partials/comments.html` (根目录而不是docs目录)，并将**启用giscus**内容将复制至注释部分

    ??? info "comments.html"
        ```html
        {% if page.meta.comments %}
        <h2 id="__comments">{{ lang.t("meta.comments") }}</h2>
        <!-- Insert generated snippet here -->

        <!-- Synchronize Giscus theme with palette -->
        <script>
            var giscus = document.querySelector("script[src*=giscus]")

            /* Set palette on initial load */
            var palette = __md_get("__palette")
            if (palette && typeof palette.color === "object") {
            var theme = palette.color.scheme === "slate" ? "dark" : "light"
            giscus.setAttribute("data-theme", theme) 
            }

            /* Register event handlers after documented loaded */
            document.addEventListener("DOMContentLoaded", function() {
            var ref = document.querySelector("[data-md-component=palette]")
            ref.addEventListener("change", function() {
                var palette = __md_get("__palette")
                if (palette && typeof palette.color === "object") {
                var theme = palette.color.scheme === "slate" ? "dark" : "light"

                /* Instruct Giscus to change theme */
                var frame = document.querySelector(".giscus-frame")
                frame.contentWindow.postMessage(
                    { giscus: { setConfig: { theme } } },
                    "https://giscus.app"
                )
                }
            })
            })
        </script>
        {% endif %}
        ```



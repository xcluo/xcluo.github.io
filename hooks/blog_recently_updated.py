"""
MkDocs 钩子：为博客首页生成最近更新列表
"""
from pathlib import Path
from datetime import datetime, date as date_type
import yaml
import re
import html as html_module


def on_page_content(html, page, config, files):
    """
    在页面内容生成时，为博客首页添加最近更新列表
    """
    # 只处理博客首页
    if "blog/index.md" not in page.file.src_path and "blog\\index.md" not in page.file.src_path:
        return html

    # 获取博客帖子目录
    blog_posts_dir = Path(config["docs_dir"]) / "blog" / "posts"

    if not blog_posts_dir.exists():
        print(f"[Blog Hook] Blog posts directory not found: {blog_posts_dir}")
        return html

    # 收集所有博客帖子及其日期信息
    posts = []
    for md_file in blog_posts_dir.glob("*.md"):
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 解析 frontmatter
        try:
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1]) or {}

                # 获取日期（优先从 frontmatter 读取）
                date_info = frontmatter.get("date", {})
                if isinstance(date_info, dict):
                    update_date = date_info.get("updated") or date_info.get("created")
                else:
                    update_date = date_info

                # 解析日期对象
                date_obj = None
                date_str = ""

                if update_date:
                    # YAML 解析器会将日期字符串自动转换为 date 对象
                    if isinstance(update_date, date_type):
                        date_obj = update_date
                        date_str = update_date.strftime("%Y-%m-%d")
                    elif isinstance(update_date, str):
                        try:
                            date_obj = datetime.strptime(update_date, "%Y-%m-%d")
                            date_str = update_date
                        except ValueError:
                            date_obj = None
                            date_str = ""

                # 如果 frontmatter 中没有日期，尝试从 document-dates 插件的缓存中获取
                # document-dates 插件会将日期存储在 page.meta["_mx"]["document_dates"]["dates"] 中
                # 但由于 on_page_content 钩子无法直接访问其他页面的 meta，需要从文件修改时间回退
                if not date_obj:
                    # 使用文件的修改时间作为回退（与 document-dates 插件的 fallback_to_file_date 行为一致）
                    file_mtime = datetime.fromtimestamp(md_file.stat().st_mtime)
                    date_obj = file_mtime.date()
                    date_str = date_obj.strftime("%Y-%m-%d")
                    print(f"[Blog Hook] Using file mtime for {md_file.name}: {date_str}")

                # 获取标题
                title = frontmatter.get("title", md_file.stem)

                # 获取 slug（用于生成 URL）
                slug = frontmatter.get("slug", md_file.stem)
                # slug 中的下划线会被 blog 插件转换为连字符
                slug = slug.replace("_", "-")

                # 根据 blog 插件配置生成正确的 URL 路径
                # post_url_format: "{date}/{slug}" 和 post_url_date_format: yyyy/MM/dd
                # use_directory_urls: false 表示 URL 需要 .html 后缀
                if date_obj:
                    # 日期格式：yyyy/MM/dd (2026/07/29)
                    date_path = date_obj.strftime("%Y/%m/%d")
                    # URL 格式：blog/2026/07/29/slug.html
                    url = f"{date_path}/{slug}.html"
                else:
                    # 如果没有日期，使用默认路径
                    url = f"posts/{slug}.html"

                posts.append({
                    "title": title,
                    "url": url,
                    "date": date_obj,
                    "date_str": date_str
                })
        except Exception as e:
            print(f"[Blog Hook] Warning: Failed to parse {md_file}: {e}")
            continue

    # 按日期降序排序（最近的在前）
    posts.sort(key=lambda x: x["date"] or datetime.min, reverse=True)

    # 生成 HTML（对标题进行 HTML 转义以防止 XSS）
    recently_updated_html = '''
<div class="recently-updated">
  <h2>最近更新</h2>
  <ul class="task-list">
'''

    for post in posts:
        date_display = post["date_str"] or "未指定日期"
        # 对标题进行 HTML 转义
        safe_title = html_module.escape(post["title"])
        recently_updated_html += f'''    <li>
      <a href="{post["url"]}" title="{safe_title}">{safe_title}</a>
      <span class="post-date">{date_display}</span>
    </li>
'''

    recently_updated_html += '''
  </ul>
</div>
'''

    # 将最近更新模块插入到页面中
    # 在第一个 h1 标签之后插入
    match = re.search(r'(</h1>)', html)
    if match:
        insert_pos = match.end()
        html = html[:insert_pos] + recently_updated_html + html[insert_pos:]
    else:
        print(f"[Blog Hook] Could not find h1 tag to insert content")

    return html

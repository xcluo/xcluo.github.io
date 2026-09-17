"""
MkDocs 钩子：为博客首页生成最近更新列表
使用 Git commit 时间进行排序，而非文件修改时间
"""
from pathlib import Path
from datetime import datetime, date as date_type
import yaml
import re
import html as html_module
import subprocess
import os


def get_git_last_updated_dates(docs_dir_path: Path) -> dict:
    """
    获取所有 .md 文件的 Git 最后提交时间
    返回格式: {相对路径: 时间戳}
    """
    doc_mtime_map = {}
    try:
        # 获取 Git 仓库根目录
        git_root = Path(subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=docs_dir_path, encoding='utf-8'
        ).strip())
        
        # docs 目录相对于 git 根目录的路径
        rel_docs_path = docs_dir_path.relative_to(git_root).as_posix()

        # 获取每个 .md 文件的最后提交时间（不使用 --relative，让 git 返回完整路径）
        cmd = ['git', 'log', '--no-merges', '--format=%at', '--name-only', 
               '--', '*.md']
        process = subprocess.run(cmd, cwd=docs_dir_path, capture_output=True, encoding='utf-8')
        
        if process.returncode == 0:
            # 获取已跟踪的文件列表（相对于 docs 目录）
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=docs_dir_path, capture_output=True, encoding='utf-8'
            )
            # 过滤出 .md 文件，并确保路径是相对于 docs 的
            tracked_files = set()
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.endswith('.md'):
                    # ls-files 返回的是相对于 cwd 的路径，已经是相对路径
                    tracked_files.add(line)
            
            ts = None
            for line in process.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                # 时间戳行格式: 1234567890
                if line.isdigit():
                    ts = float(line)
                # 文件路径行（带 docs/ 前缀）
                elif line.endswith('.md') and ts:
                    # 去掉 docs/ 前缀（如果存在）
                    rel_path = line
                    if rel_path.startswith(rel_docs_path + '/'):
                        rel_path = rel_path[len(rel_docs_path) + 1:]
                    elif rel_path.startswith(rel_docs_path):
                        rel_path = rel_path[len(rel_docs_path):].lstrip('/')
                    
                    if rel_path in tracked_files:
                        # 使用 setdefault，只记录第一次出现的文件（即最近一次提交）
                        doc_mtime_map.setdefault(rel_path, ts)
                    ts = None  # 重置，等待下一个文件
    except Exception as e:
        print(f"[Blog Hook] Error getting git info: {e}")
    
    return doc_mtime_map


def on_page_content(html, page, config, files):
    """
    在页面内容生成时，为博客首页添加最近更新列表
    按 Git commit 时间排序
    """
    # 只处理博客首页
    if "blog/index.md" not in page.file.src_path and "blog\\index.md" not in page.file.src_path:
        return html

    # 获取博客帖子目录
    blog_posts_dir = Path(config["docs_dir"]) / "blog" / "posts"

    if not blog_posts_dir.exists():
        print(f"[Blog Hook] Blog posts directory not found: {blog_posts_dir}")
        return html

    # 获取 Git 最后提交时间（所有 .md 文件）
    git_dates = get_git_last_updated_dates(Path(config["docs_dir"]))

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

                # 获取 Git 时间（用于排序）
                # 文件在 Git 中的相对路径
                try:
                    rel_path = md_file.relative_to(Path(config["docs_dir"])).as_posix()
                except ValueError:
                    rel_path = str(md_file.name)
                
                git_timestamp = git_dates.get(rel_path)
                
                # 如果 Git 中没有记录，回退到文件 mtime
                if git_timestamp:
                    git_date = datetime.fromtimestamp(git_timestamp)
                else:
                    file_mtime = datetime.fromtimestamp(md_file.stat().st_mtime)
                    git_date = file_mtime

                # 如果 frontmatter 中没有日期，使用 Git 时间
                if not date_obj:
                    date_obj = git_date.date()
                    date_str = date_obj.strftime("%Y-%m-%d")

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
                    # URL 格式：/blog/2026/07/29/slug.html（绝对路径）
                    url = f"/blog/{date_path}/{slug}.html"
                else:
                    # 如果没有日期，使用默认路径
                    url = f"/blog/posts/{slug}.html"

                posts.append({
                    "title": title,
                    "url": url,
                    "date": date_obj,
                    "date_str": date_str,
                    "git_date": git_date  # 用于排序
                })
        except Exception as e:
            print(f"[Blog Hook] Warning: Failed to parse {md_file}: {e}")
            continue

    # 按 Git commit 时间降序排序（最近的在前）
    posts.sort(key=lambda x: x["git_date"] or datetime.min, reverse=True)

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
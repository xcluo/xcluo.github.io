"""
归档和分类页面 - 使用与"最近更新"相同的样式显示内容
通过 on_page_content 钩子替换默认的博客文章列表为紧凑的单行样式
"""
import re
import html as html_module


def on_page_content(html, page, config, files):
    """
    针对归档页面 (blog/archive/*) 和分类页面 (blog/category/*)
    使用与"最近更新"相同的样式显示文章列表
    """
    src_path = page.file.src_path.replace('\\', '/')
    
    # 检查是否是归档或分类页面
    is_archive = src_path.startswith('blog/archive/')
    is_category = src_path.startswith('blog/category/')
    
    if not (is_archive or is_category):
        return html
    
    # 获取页面标题（归档名或分类名）
    page_title = page.title if hasattr(page, 'title') else '归档'
    
    # 尝试从 context 获取 posts
    # blog 插件会在 context 中注入 posts 数据
    posts = []
    
    # 尝试从页面的 attributes 获取 posts（blog 插件会设置）
    if hasattr(page, 'posts'):
        posts = page.posts
    elif hasattr(page, 'context') and page.context and 'posts' in page.context:
        posts = page.context['posts']
    
    # 如果没有获取到 posts，尝试解析页面的 HTML 来获取文章信息
    if not posts:
        # 回退方案：从当前页面 HTML 中提取文章链接
        # 这个方法可能在不同版本中不稳定，但作为后备方案
        pass
    
    # 生成与 recently-updated 相同样式的 HTML
    archive_html = f'''
<div class="recently-updated">
  <h2>{html_module.escape(str(page_title))}</h2>
  <ul class="task-list">
'''
    
    # 如果有 posts 数据，生成文章列表
    if posts:
        for post in posts:
            # 获取文章信息
            title = getattr(post, 'title', str(post)) or '无标题'
            # 获取文章 URL，确保是绝对路径
            url = getattr(post, 'url', '#') or '#'
            if not url.startswith('/'):
                url = '/' + url
            # 获取发布日期
            date_str = ""
            if hasattr(post, 'config') and hasattr(post.config, 'date'):
                date_obj = post.config.date
                if hasattr(date_obj, 'created'):
                    date_str = date_obj.created.strftime("%Y-%m-%d")
                elif hasattr(date_obj, 'updated'):
                    date_str = date_obj.updated.strftime("%Y-%m-%d")
            
            # HTML 转义防止 XSS
            safe_title = html_module.escape(str(title))
            safe_url = html_module.escape(str(url))
            
            archive_html += f'''    <li>
      <a href="{safe_url}" title="{safe_title}">{safe_title}</a>
      <span class="post-date">{date_str}</span>
    </li>
'''
    else:
        # 如果没有 posts，显示提示信息
        archive_html += '''    <li>
      <span class="post-date">暂无文章</span>
    </li>
'''
    
    archive_html += '''
  </ul>
</div>
'''
    
    # 在页面标题后插入文章列表
    match = re.search(r'(</h1>)', html)
    if match:
        insert_pos = match.end()
        html = html[:insert_pos] + archive_html + html[insert_pos:]
    else:
        # 如果没有 h1，尝试在主要内容区域插入
        match = re.search(r'(<div class="md-content[^"]*">)', html)
        if match:
            insert_pos = match.end()
            html = html[:insert_pos] + archive_html + html[insert_pos:]
    
    return html
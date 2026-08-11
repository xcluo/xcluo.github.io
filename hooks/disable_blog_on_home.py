def on_page_context(context, page, config, nav):
    # 只针对主页（docs/index.md）
    if page.file.src_uri == 'index.md':
        # 移除博客文章列表数据（插件注入的键通常叫 'posts'）
        # 如果未来版本键名变化，可以打印 context.keys() 查看
        context.pop('posts', None)
        # 如果有分页相关数据也一并移除
        context.pop('paginator', None)
        context.pop('page_obj', None)
    return context
"""
在构建完成后复制自定义 CSS 文件到 site 目录
"""
import os
import shutil
from pathlib import Path

def on_post_build(config, **kwargs):
    """构建完全结束后复制 CSS 文件（比 on_post_build 更晚执行）"""
    print("[copy_css] Hook started (on_build_complete)")
    
    # CSS 文件列表
    css_files = [
        "packages/stylesheets/css/blog.css",
        "packages/stylesheets/css/extra_img.css",
        "packages/stylesheets/css/extra_font.css",
        "packages/stylesheets/css/extra_component.css",
        "packages/stylesheets/css/extra_code.css",
        "packages/stylesheets/css/extra_badge.css",
    ]

    site_dir = Path(config['site_dir'])

    for css_file in css_files:
        src = Path(css_file)
        dst = site_dir / css_file

        if src.exists():
            try:
                # 确保目标目录存在
                dst.parent.mkdir(parents=True, exist_ok=True)
                # 复制文件
                shutil.copy2(src, dst)
                print(f"[copy_css] Copied: {src} -> {dst}")
            except Exception as e:
                print(f"[copy_css] ERROR copying {src}: {e}")
        else:
            print(f"[copy_css] SKIPPED (file not found): {src}")
    
    print("[copy_css] Hook finished")

---
title: "Playwright"
---

- 新使用playwright电脑开启时就关闭历史网页，会出现报错，第一次应将new_page_to_start置为false即可
- 只有open了url后，才能从该页面执行cookies_get
- browser_utils.py的evaluate方法中，增加*args参数

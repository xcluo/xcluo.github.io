---
draft: true 
date: 2026-01-23
title: "browser_use"
comments: true
categories:
  - AI应用
---



<!-- more -->


---

# browser_use tool

## 使用自定义浏览器

- COPAW_BROWSER_USE_DEFAULT:0
- PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH:C:\Users\LXC\AppData\Local\Google\Chrome\Application\chrome.exe

## action

- action: {start, stop, open, navigate, navigate_back, snapshot, screenshot, click, type, eval, evaluate, resize, console_messages, network_requests, handle_dialog, file_upload, fill_form, install, press_key, run_code, drag, hover, select_option, tabs, wait_for, pdf, close, cookies_get, cookies_set, cookies_clear, connect_cdp, list_cdp_targets, clear_browser_cache}
	- start：启动浏览器，常搭配header与cdp_port参数使用，如{"action": "start", "headed": true}
	- open：创新新标签页并打开网址，常搭配url参数使用
	- navigate：在当前页面跳转至网址，常搭配url参数使用
	- screenshot：截图保存，获取视觉图像（像素级），常搭配path和full_page使用
	- snapshot：抓取页面结构，获取可交互元素（DOM 分析），生成带引用id的元素树，常搭配frame_selector和snapshot_filename参数使用，前者指定iframe，后者指定快照数据存放位置
	> snapshot每次抓取都不一定一样，因此多次修改后建议重新抓取
	- click：点击元素，常搭配ref、selector和wait参数使用，
		- ref：snapsho元素引用id
		- selector：CSS选择器
		- button：{left, right, middle}分别点击鼠标左、右及中键
	- eval：执行js简化版代码
	- evaluate：执行js完整版代码，可获取返回值
	- run_code：运行更长的代码块或脚本
	- stop：关闭浏览器
	- cookies_get：获取指定页面cookie信息，{"action": "cookies_get", "url": "http://10.180.253.162:30005/knowledge/document"} 

跳转操作常用wait（单位为ms）参数等待页面刷新

### CSS选择器基本语法

| 选择器类型 | 语法 | 示例 | 说明 |
| :--- | :--- | :--- | :--- |
| 元素选择器 | `element` | `div`, `p` | 选中所有指定的 HTML 标签。 |
| 类选择器 | `.classname` | `.btn`, `.header` | 选中所有 `class="..."` 属性包含指定值的元素。 |
| ID 选择器 | `#idname` | `#main`, `#logo` | 选中 `id="..."` 的唯一元素。 |
| 通配符选择器 | `*` | `*` | 选中页面上的所有元素。 |
| 群组选择器 | `A, B` | `h1, p, .title` | 同时选中多个选择器，为它们设置相同的样式。 |

元素：div，p，button，a，input

- class属性命名风格：`<元素 class="类名1 类名2 类名3">`，每个类由空格隔开
- 元素选择器：`div`，`p`，div内部的p `div.p`
- 属性选择器（[]前可用搭配元素用于约束范围，如`img[attr=value]`）：含属性`[attr]`，具体值`[attr=value]`，以value开头`[attr^=value]`，以value结尾`[attr$=value]`，含value内容`[attr*=value]`
- 组合选择器：直接子元素 `div > p`，相邻兄弟元素 `p + p`，通用兄弟 `p ~ p`，任意层级所有后代 `div p`

## headed

false，是否为可见窗口模式

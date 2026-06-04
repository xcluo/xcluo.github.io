---
comments: true
title: 北信源卸载
date: 2025-01-31 
slug: xcluo
---

<!-- more -->

- 服务中关闭V开头的服务
- 任务管理器关闭相关.exe进程(一般包含关键字v)，如csvr.exe
- 删除各盘符中的【隐藏含VR的文件夹及CEMS文件夹】
- 使用geek右键强制删除北信源
- 使用火绒右键强制粉碎剩余安装文件
- 使用命令强制删除`rm -rf path_to_VRV`
- 重启后重复上述步骤，直至再无path_to_VRV安装文件夹
- 
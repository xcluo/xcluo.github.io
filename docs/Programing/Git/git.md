---
title: Git
---

### 基本概念

<div class="one-image-container">
    <img src="image/repository_related.jpg" style="width: 80%;">
    <!-- <p>LoRA在Attention各部分权重上的消融实验效果</p> -->
    <!-- <figcaption>这是图片的标题或描述。</figcaption> -->
</div>

- 未跟踪区：Untracked
- 工作区：Workspace，Unstaged
- 暂存区：Staged，Index
- （本地）仓库区：Repository
- 远程仓库区：Remote

### 基本操作

- `git status`

## 仓库操作

### Untracked ⟷ Unstaged ⟷ Staged

#### `git add`

将工作区更新或未跟踪的文件添加到暂存区

```bash
git add file.txt
git add .               # 将当前目录下所有文件及文件夹添加至暂存区
```

#### `git restore`

把文件恢复成暂存区（或最后一次提交）的样子，丢弃工作区的所有改动，基本语法为 `git restore [OPTION] file_name`

Option

- `--staged` 只操作暂存区
- `--worktree` 默认操作，即对工作区和暂存区均进行改动，可不用显示写出

```bash
git restore file.txt                # 直接重置改动，重置为上一次Staged状态（无Staged状态则为最新一次提交状态HEAD）
git restore --staged file.txt       # 保留改动，只将文件由暂存区 → 工作区

# Staged → Unstaged → HEAD
git restore --staged --worktree file.txt
```

!!! info
    - `git restore` 默认操作Unstaged和Staged区域，不对Untracked区域做处理
    - `--staged` 只对暂存区文件做移出处理

#### `git rm`

将指定文件从暂存区移除（变为Untracked），基本语法为`git rm [OPTION] file_name`

Option

- `-f` --force，强制处理
- `-r` 递归处理文件夹中内容
- `--cached` 从暂存区中取消缓存，不物理删除文件
- `-n` --dry-run，不做任何移除操作，只是显示执行该命令有哪些文件会被从暂存区中移除

```bash
# rm + git add物理删除并提交删除记录
git rm file.txt

# 暂存区 → untracked
git rm --cached file.txt
```

#### `git clean`

用于删除未跟踪的文件和文件夹，基本语法为 `git clean [OPTION] file_name`

Option

- `-n` --dry-run 预览模式，仅显示什么文件会被删
- `-f` --force 强制删除，不加该参数无法真正实现删除
- `-d` 删除文件夹
- `-x` 连 ignored 文件一起删除
- `-X` 只删 ignored 文件，保留其他未跟踪文件
- `-i` --interactive 交互式，逐个删除确认

> ignored文件指的是.gitignore中指定的文件

```bash
git clean
git clean -df
```

#### `git reset`

将暂存区的文件还原到工作区，将暂存区指定文件状态回退

| 模式 | Index | WorkSpace | 危险性 |
| :---: | :---: | :---: | :---: |
| `--soft` | 不改变 | 不改变 | 低 |
| `--mixed` | 重置 | 不改变 | 中 |
| `--hard` | 重置 | 重置 | 高 |

```bash
<!-- 将缓存区(指定文件)回退至指定版本提交状态 -->
git reset (--mixed) (HEAD)
git reset HEAD~n                        # 回退至上n个版本
git reset <commit-hash>
git reset <commit-hash> <file_name>     # 撤销文件缓存
```

### Staged ⟷ 本地仓库

#### `git commit`

将暂存区的改动永久保存到本地仓库，形成一条可追溯的历史记录。基本语法为 `git commit [OPTION] [commit message]`

Option

- `-m "commit message"`
- `-a` --all，自动暂存所有已跟踪文件的修改
- `--amend` 修改上一次提交（改commit message或补文件）
- `--no-verify` 强制提交，跳过pre-commit检查等钩子

```bash
git commit -am "commit message"         # 快速提交已跟踪文件改动

# --amend # 
git commit --amend -m "commit message"  # 修改最近一次提交的commit message & 补文件
git commit --amend --no-edit            # 补文件

git commit --no-verify -m "commit message"
                                        # 强制提交，跳过检查
```

!!! info
    如果最新一次提交已经推送到远程仓库：
    1. 本地使用 `--amend` 修改提交信息后，的本地仓库SHA就和远程仓库SHA不一致，导致冲突
    2. 需使用 `git push --force` 或更安全的 `git push --force-with-lease` 来强制推送，用本地的新提交覆盖远程的旧提交。

    > 使用 `git push --force-with-lease` 需--amend前的HEAD与远程仓库HEAD一致才可成功

### 本地仓库 ⟷ 远程仓库

#### `git push`

#### `git pull`

#### `git clone`

拉取指定分支

`git clone -b <branch_name> <git_url of SSH/HTTP>`

## 分支操作

#### `git branch`

```bash
git branch                      # 查看本地分支信息
git branch -r                   # 查看远程分支信息
git branch -a                   # 查看本地分支和远程分支
git branch -m <new_branch_name> # 重命名当前分支
git branch -M <new_branch_name> # 重命名当前分支并强制覆盖已有分支（若重名）

git branch <new_branch_name>    # 创建但不切换至新分支

# 删除本地分支, 不能删除当前分支
git branch -d branch_name       # 普通删除分支
git branch -D branch_name       # 强制删除分支

# 删除远程分支
git push origin --delete remote_branch_name

git branch --vv                 # 查看本地分支和远程分支关联情况
git branch --unset-upstream     # 解除当前本地分支远程分支的关联
git branch --set-upstream-to=origin/<new_remote_branch_name>
git branch -u origin/<new_remote_branch_name>
                                # 将当前本地分支关联到新的目标远程分支
```

#### `git checkout`

```bash
# 以parent_branch_name为父分支生创建新的本地分支new_branch_name
# origin/remote_branch_name，使用远程分支作为父分支
git checkout -b <new_branch_name> <parent_branch_name>
git checkout -b <branch_name> <sha> # 将某次提交结果作为新分支并创建新分支
git checkout -b <new_branch_name>   # 创建并切换至新分支，默认父分支为当前分支
git checkout <branch_name>          # 切换至指定分支
git checkout -                      # 切换至上一分支
```

#### `git merge`

```bash
# 将目标分支合并至当前分支
git merge <merged_branch_name>
git merge --no-ff <merged_branch_name>  # ff: fast-forward，默认参数值
git merge --abort                       # 合并时发生冲突，终止合并
```

> 修改完冲突后, 通过add冲突文件 + `git commit`即可继续完成合并

#### `git diff`

- `git diff hash_1 hash_2`

#### `rebase`

```bash
git rebase -i HEAD~n                    # 将最近n次提交合并为1次
                                        # HEAD为最新提交，HEAD~n为最近的前n次提交
                                        # HEAD~n，实际上包括n+1次提交

```

### 日志相关

### 配置文件

### 常用git仓库

1. github  
2. gitlab

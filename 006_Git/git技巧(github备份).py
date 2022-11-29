# Databricks notebook source
# MAGIC %md
# MAGIC ### (2022.7.28更新)
# MAGIC 
# MAGIC 返回上级目录
# MAGIC `cd ..`
# MAGIC 
# MAGIC 进入到某级目录
# MAGIC `code xx`
# MAGIC 
# MAGIC 显示当前路径
# MAGIC `pwd`
# MAGIC 
# MAGIC 清屏
# MAGIC `clear`
# MAGIC 
# MAGIC 列出当前路径下文件
# MAGIC `ls`
# MAGIC 
# MAGIC 列出文件的详细信息
# MAGIC `ls -l`
# MAGIC 
# MAGIC 列出路径下所有的文件，包含隐藏文件
# MAGIC `ls -a`
# MAGIC 
# MAGIC 创建一个文件夹目录
# MAGIC `mkdir xx`
# MAGIC 
# MAGIC 创建一个新文件
# MAGIC `touch xx.xx`
# MAGIC 
# MAGIC 移除一个文件
# MAGIC `rm xx.xx`
# MAGIC 
# MAGIC 移除一个文件夹
# MAGIC `rm -r xx`
# MAGIC 
# MAGIC 强制移除某个文件或文件夹
# MAGIC ```
# MAGIC rm -rf xx (never use "rm -rf", will delete everthing.)
# MAGIC ```
# MAGIC 
# MAGIC * x1文件,x2文件夹：移动文件x1到文件夹x2\n'
# MAGIC '* x1文件夹,x2不存在,重命名文件夹x1为x2\n'
# MAGIC '* x1文件夹,x2文件夹,将文件夹x1移动到文件夹x2内
# MAGIC `mv x1 x2`
# MAGIC 
# MAGIC 显示所有命令记录历史
# MAGIC `history`
# MAGIC 
# MAGIC 显示帮助文档
# MAGIC `help xx`
# MAGIC 
# MAGIC 退出bash环境
# MAGIC `exit`
# MAGIC 
# MAGIC 显示当前git配置信息
# MAGIC `git config -l`
# MAGIC 
# MAGIC 显示当前系统的git配置信息
# MAGIC ```
# MAGIC git config --system --list
# MAGIC ```
# MAGIC 
# MAGIC 显示全局配置的内容，用户名+账户等，必须配置
# MAGIC ```
# MAGIC git config --global --list
# MAGIC ```
# MAGIC 
# MAGIC * git系统配置文件\n'
# MAGIC '* git全局配置文件路径
# MAGIC ```
# MAGIC c:\program\files\git\etc\gitconfig\nc:\.gitconfig
# MAGIC ```
# MAGIC 
# MAGIC 设置用户名和邮箱-用户标识
# MAGIC ```
# MAGIC git config --global --user.name "xx"\ngit config --global --user.email "xx"
# MAGIC ```
# MAGIC 
# MAGIC 显示git常用的命令
# MAGIC `git`
# MAGIC 
# MAGIC 查看之前的版本历史信息
# MAGIC `git log`
# MAGIC 
# MAGIC 简化git版本历史输出的信息
# MAGIC ```
# MAGIC git log --pretty=online
# MAGIC ```
# MAGIC 
# MAGIC 在当前路径初始化一个git项目
# MAGIC `git init`
# MAGIC 
# MAGIC 显示当前路径下所有文件的git状态
# MAGIC `git status`
# MAGIC 
# MAGIC 将路径下所有文件添加到stage区
# MAGIC `git add .`
# MAGIC 
# MAGIC 提交stage区的内容到本地仓库(如报错检查gitconfig信息)
# MAGIC ```
# MAGIC git commit -m "<commit.message>"
# MAGIC ```
# MAGIC 
# MAGIC 将本地仓库内容同步到远程仓库
# MAGIC `git push`
# MAGIC 
# MAGIC 将远程仓库克隆到本地
# MAGIC `git clone <url>`
# MAGIC 
# MAGIC 将最新的远程仓库同步到本地
# MAGIC `git pull`
# MAGIC 
# MAGIC git忽略本地项目更改，直接用服务器版本来覆盖本地仓库
# MAGIC ```
# MAGIC git fetch --all\ngit reset --hard origin/master\ngit pull
# MAGIC ```
# MAGIC 
# MAGIC 显示当前项目所有分支
# MAGIC `git branch`
# MAGIC 
# MAGIC 显示所有远程分支
# MAGIC `git branch -r`
# MAGIC 
# MAGIC 新建一个分支并停留在当前分支
# MAGIC `git branch <branch.name>`
# MAGIC 
# MAGIC 新建一个分支并切换到该分支上
# MAGIC `git checkout -b <branch.name>`
# MAGIC 
# MAGIC 转换工作区分支到新创建的分支上
# MAGIC `git switch <branch.name>`
# MAGIC 
# MAGIC 如果remote仓库没有这个branch，则这个分支直接git push会无效，可以用这个命令来实现分支push
# MAGIC ```
# MAGIC git push --set-upstream origin <branch.name>
# MAGIC ```
# MAGIC 
# MAGIC 合并指定分支到当前分支上
# MAGIC `git merge <branch.name>`
# MAGIC 
# MAGIC 删除远程分支
# MAGIC ```
# MAGIC git push origin --delete <branch.name>\ngit branch -dr <remote/branch>
# MAGIC ```
# MAGIC 
# MAGIC 显示当前路径下所有远程库
# MAGIC `git remove -v`
# MAGIC 
# MAGIC 查看本机是否安装ssh
# MAGIC `ssh`
# MAGIC 
# MAGIC 生成ssh公钥
# MAGIC `ssh-keygen`
# MAGIC 
# MAGIC 使用加密算法生成公钥，一路回车;c:/admin/.ssh下回生成两个文件, .pub后缀的文件打开后，与Gitlab/Github绑定
# MAGIC `ssh-keygen -t ras`
# MAGIC 
# MAGIC git端验证绑定ssh是否成功
# MAGIC ```
# MAGIC ssh -T git@github.com | ssh -T git@gitlab.com ..
# MAGIC ```
# MAGIC 
# MAGIC 输出所有git提交历史日志，查找要退回的版本
# MAGIC `git log`
# MAGIC 
# MAGIC 进行版本回退操作
# MAGIC `git reset --hard <commit_id>`
# MAGIC 
# MAGIC 回滚后推送至远程分支
# MAGIC `git push origin`
# MAGIC 
# MAGIC 快捷命令，回退到上个版本
# MAGIC ```
# MAGIC git reset --hard HEAD^ | HEAD^^-previous~previous
# MAGIC ```
# MAGIC 
# MAGIC 进入python交互模式
# MAGIC `python -i`
# MAGIC 
# MAGIC 进入python终端命令行交互模式
# MAGIC `winpty python`
# MAGIC 
# MAGIC 退出终端python交互模式
# MAGIC `quit()`
# MAGIC 
# MAGIC bash中运行.py文件，运行完成后停留在python终端环境
# MAGIC `python -i xx.py`
# MAGIC 
# MAGIC bash中运行.py文件，运行完成后退出python终端环境
# MAGIC `winpty python xx.py`
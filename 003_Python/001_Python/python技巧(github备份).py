# Databricks notebook source
# MAGIC %md
# MAGIC ### (2022.7.28更新) | Python Coding笔记
# MAGIC 
# MAGIC 1. 一键打开经常用的多个网页
# MAGIC ```import webbrowser
# MAGIC websites = [
# MAGIC     'www.google.com',
# MAGIC     'www.bing.com',
# MAGIC     'www.yahoo.com'
# MAGIC ]
# MAGIC for url in websites:
# MAGIC     webbrowser.open(url)
# MAGIC ```
# MAGIC 
# MAGIC 2. 在服务器中定位当前绝对路径
# MAGIC ```
# MAGIC import os
# MAGIC current_path = os.path.abspath('<file.name.xxx>')
# MAGIC print(current_path)
# MAGIC ```
# MAGIC 
# MAGIC 3. pip批量安装依赖第三方库
# MAGIC ```
# MAGIC create 'requirement.txt' file
# MAGIC file format like 'pandas==0.1.1'
# MAGIC then run 'pip install -r requirements.txt' in terminal
# MAGIC ```
# MAGIC 
# MAGIC 4. pip清华镜像
# MAGIC ```
# MAGIC pip install -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
# MAGIC ```
# MAGIC 
# MAGIC 5. streamlit默认设置宽屏 
# MAGIC `st.set_page_config(layout='wide')`
# MAGIC 
# MAGIC 1. 为python大型新项目创建虚拟环境（虚拟环境的好处：相当于容器docker，在这个容器中，我们可以只安装项目中需要的依赖包，各个容器之间互相隔离，互不影响）:
# MAGIC     - 安装虚拟环境库 `pip install virtualenv`, 选择目标路径文件夹 `cd xxx/envname`，在目标路径下创建虚拟环境 `virtualenv envname`. 
# MAGIC     - 命令行启动虚拟环境。找到activate文件所在路径，有在bin(可能没有)，有时在envname/Scripts文件夹中，直接在命令行中输入activate文件所在路径即可启动虚拟环境 `C:/xxxx/xx>.../envname/Scripts/activate`，启动后终端变为`(envname) C:/xxxx...`，退出同理： `C:/xxxx/xx>.../envname/Scripts/deactivate`
# MAGIC 1. vscode中如何配置及使用python虚拟环境:
# MAGIC     - 直接像在终端中激活虚拟环境一样操作会报错，是‘vscode执行策略’的原因导致的。解决方案如下：
# MAGIC     - 更改vscode运行权限为管理员权限，属性-兼容性-以管理员身份运行此程序
# MAGIC     - 在vscode终端执行命令 `set-executionpolicy remotesigned`，再执行`get-executionpolicy`，执行结果显示 `remotesigned` 即可。默认情况下会显示 `remoterestricted`
# MAGIC     - 在vscode终端找到虚拟环境启动路径，输入./`activate` 启动虚拟环境，输入 `deactivate` 退出虚拟环境
# MAGIC     - 在vscode中的jupyter notebook中使用虚拟环境，在vscode-jupyter界面的右上角，选择-更改jupyter内核，可以看到虚机路径的选项，点击选择对应vm即可（如有必要，点击`ctrl+shift+p`，在`python：select interpreter` 中选择对应虚拟环境中的python.exe，选项前会有个五角星`☆`的标志）
# MAGIC     - 导出虚拟环境，在虚拟环境目录中：`pip freeze > environment.yml`
# MAGIC     - 安装虚拟环境，在目标路径中：`pip install -r environment.yml`
# MAGIC 1. vscode 中使用 git 同步项目：
# MAGIC     - 直接调出 vscode 命令行（ctrl+~），vscode 会自动识别在文件夹中的git文件。直接使用 `git add .`, `git commit -m 'x'`, `git push` 等命令 (如此就省去了在git bash中同步仓库文件的方法)
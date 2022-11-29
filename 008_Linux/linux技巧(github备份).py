# Databricks notebook source
# MAGIC %md
# MAGIC # Linux (Raspbian)
# MAGIC `shell`
# MAGIC 
# MAGIC 更新密码
# MAGIC `'sudo passwd pi'`
# MAGIC 
# MAGIC 调整vnc屏幕分辨率
# MAGIC `vncserver -geometry 1080*720`
# MAGIC 
# MAGIC 打开vnc配置菜单
# MAGIC `sudo raspi-config`
# MAGIC 
# MAGIC 给树莓派安装外设屏幕驱动
# MAGIC ```
# MAGIC git clone https://github.com/waveshare/LCD-show.git
# MAGIC cd LCD-show/
# MAGIC sudo ./LCD35-show
# MAGIC ```
# MAGIC 
# MAGIC 显示当前温度
# MAGIC `vcgencmd measure_temp`
# MAGIC 
# MAGIC 查找树莓派ip地址
# MAGIC `ping raspberrypi.local -4`
# MAGIC 
# MAGIC 检查树莓派是否连接wifi成功
# MAGIC `ifconfig wlan0`
# MAGIC 
# MAGIC 为树莓派添加新wifi
# MAGIC `
# MAGIC sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
# MAGIC `
# MAGIC 
# MAGIC 在文件底部添加新wifi信息
# MAGIC ```
# MAGIC network={
# MAGIC ssid="The SSID of your network (eg. Network name)"
# MAGIC psk="Your Wifi Password"
# MAGIC }
# MAGIC ctrl x - y 保存后退出
# MAGIC ```
# MAGIC 
# MAGIC 显示当前文件路径
# MAGIC `pwd`
# MAGIC 
# MAGIC 查看目录中的全部文件
# MAGIC `ls`
# MAGIC 
# MAGIC 显示文件和目录中全部信息
# MAGIC `ls -l`
# MAGIC 
# MAGIC 列出全部文件，包含隐藏文件
# MAGIC `ls -a` 
# MAGIC 
# MAGIC 递归显示所有文件
# MAGIC `ls -R`
# MAGIC 
# MAGIC 显示这个文件所有内容
# MAGIC `cat <file.name>`
# MAGIC 
# MAGIC 获取网卡配置和系统的网络信息
# MAGIC `ifconfig`
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC 安装python3.x</br>
# MAGIC `sudo apt install python3`</br>
# MAGIC `sudo apt remove python` #卸载python2.x
# MAGIC 
# MAGIC 
# MAGIC `sudo apt autoremove` #自动清理python2.x依赖
# MAGIC 
# MAGIC 
# MAGIC `sudo rm /usr/bin/python` #删掉原来python链接
# MAGIC 
# MAGIC 
# MAGIC `sudo ls -s /usr/bin/python3.7 /usr/bin/python` #创建一个新的链接
# MAGIC 
# MAGIC 
# MAGIC `python` #输入后自动进入python终端	
# MAGIC 
# MAGIC #linux文本编辑器 (nano/vim(是vi的增强版)/vi)
# MAGIC 
# MAGIC 
# MAGIC `nano` <file.name> #打开这个文件，并显示内容
# MAGIC 
# MAGIC 
# MAGIC `ctrl x` #退出 	
# MAGIC 
# MAGIC `vi <fine.name>` #通过vi编辑器打开文件
# MAGIC 
# MAGIC `:q` #退出编辑
# MAGIC 
# MAGIC `:q!` #强制退出
# MAGIC 
# MAGIC #树莓派默认的vi，没有颜色区分，使用上很不便捷
# MAGIC `sudo apt-get install vim #安装vim`
# MAGIC 
# MAGIC #更新raspbian系统
# MAGIC `sudo apt-get update`
# MAGIC 
# MAGIC #安装更新
# MAGIC `sudo apt-get upgrade
# MAGIC `
# MAGIC 
# MAGIC #树莓派安装mysql
# MAGIC `sudo apt-get install sqlite3`
# MAGIC 
# MAGIC `sudo apt-get install libsqlite3-dev` #安装sqlite3编译时需要的工具包
# MAGIC 
# MAGIC `sqlite3 -version` #检查sqlite3版本
# MAGIC 
# MAGIC `sqlite3` #输入进入sqlite环境
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC #树莓派安装spark
# MAGIC `wget https://www.apache.org/dyn/closer.lua/spark/spark-3.2.0/spark-3.2.0-bin-hadoop3.2.tgz` #下载安装包方法一
# MAGIC 
# MAGIC `curl -O https://www.apache.org/dyn/closer.lua/spark/spark-3.2.0/spark-3.2.0-bin-hadoop3.2.tgz` #下载安装包方法二
# MAGIC 
# MAGIC 类似于wget/curl这种直接在linux系统中的下载方式，没有经过某些安装协议的要求。所以在linux环境汇中通过`tar -zxvf <file.name>.tgz` 的方式解压一直失败（后来发现这种方式下载的文件也不完整，只有几百k，但实际tgz文件有287MB）
# MAGIC 
# MAGIC 解决办法，在windows系统中下载好，然后上传到linux服务器，再进行解压缩和安装操作
# MAGIC `tar zxvf spark-3.2.0-bin-hadoop3.2.tgz`
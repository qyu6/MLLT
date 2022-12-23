###########################
# Create: 2022/12/8
# Author: T
# Purpose: Index apps
###########################


import streamlit as st
import pandas as pd
import numpy as np

st.sidebar.info('IndexPortal [T]')

choice = st.sidebar.selectbox('Category',('Work','Study','Research','Tools','Entertainment'))

# work **********************************
if choice == 'Work':
    st.markdown('Work')

    list_1 = [
        '- Azure站点 | https://portal.azure.cn/#home',
        '- 高级数据可视化方案d3js | https://d3js.org/',
        '- Agile Data Science电子书 | https://edwinth.github.io/ADSwR/index.html',
        '- 内燃机学报 | http://nrjxb.paperonce.org/',
        '- 人大统计之都 | https://cosx.org/',
        '- 麦肯锡McKinsey | https://www.mckinsey.com/',
        '- 智源社区AI平台 | https://hub.baai.ac.cn/',
        '- GE数字化平台 | https://www.ge.com/digital/',
        '- (KDD) ACM的知识发现及数据挖掘专委会 | https://kdd.org/',
        '- 微软爱写作 (英文论文自动修改) | https://aimwriting.mtutor.engkoo.com/',
    ]

    for i in list_1:
        st.markdown(i)




# study **********************************
elif choice == 'Study':
    st.markdown('Study')

    list_2 = [
        '- Pip Tsinghua国内镜像源 | https://mirrors.tuna.tsinghua.edu.cn/help/pypi/',
        '- Github | https://github.com/qyu6/ (qyu6, tqin0411@outlook.com)',
        '- Github (homepage) | https://github.com/TonyQin0/ (TonyQin0, )',
        '- Github (other) | https://github.com/xx/ (tqthooo2021@163.com)',
        '- Databricks社区版 | https://community.cloud.databricks.com/login.html/ (tqin0411@outlook.com)', 
        '- Databricks用户社区 | https://community.databricks.com/s/login/ (tqin0411@outlook.com)',
        '- Kaggle | https://www.kaggle.com/ (tqin0411@outlook.com)',
        '- Kaggle Top解决方案 | https://farid.one/kaggle-solutions/',
        '- Coding学习平台(numpy, scipy..) | https://www.runoob.com/',
        '- Shiny-for-Rstudio交互式app开发 | https://shiny.rstudio.com/',
        '- Shiny-for-Python交互式app开发 | https://shiny.rstudio.com/py/',
        '- 公开数据集下载 | https://data.mendeley.com/research-data/?search=',
        '- 统计学理论等，代码工具书电子书等下载 (中英文资源特别多) | https://bookdown.org/home/archive/',
        '- Shiny app部署 | https://www.shinyapps.io/',
        '- Azure AI Gallery开源模型库onnx | https://gallery.azure.ai/models',
        '- Markdown文档官方教程 | https://markdown.com.cn/',
        '- Streamlit部署 | https://streamlit.io/cloud/ (connect Github account)',
        '- Scikit-learn | https://scikit-learn.org/stable/user_guide.html',
        '- Keras | https://keras.io/api/ ',
        '- PyTorch | https://pytorch.org/tutorials/',
        '- TensorFlow | https://tensorflow.google.cn/',
        '- Python package发布平台|https://pypi.org/',
        '- Rstudio平台 | https://www.rstudio.com/',
        '- 图数据库neo4j | https://neo4j.com/',
        '- 中文开放知识图谱KG | http://openkg.cn/',
        '- 异常值诊断资料 | https://github.com/yzhao062/anomaly-detection-resources',


    ]

    for i in list_2:
        st.markdown(i)







# tools **********************************
elif choice == 'Tools':
    st.markdown('Tools')

    list_3 = [
        '- 在线画思维导图 | https://www.iodraw.com/mind',
        '- 115网盘 | https://115.com/',
        '- 数据可视化平台Qilk | https://www.qlik.com/us',
        '- 在线制作词云图 | https://wordart.com/create',
        '- 在线OCR识别器 | https://onlineocrconverter.com/',
        '- 一些破解软件下载站点 | https://www.jiaochengzhijia.com/sitemap.html/',
        '- EndNote x9破解版下载 | https://www.jiaochengzhijia.com/down/87119.html/',
        '- Online LaTex公式调试器 | https://latex.codecogs.com/eqneditor/editor.php?lang=zh-cn/',
        '- Markdown文本logo/符号装饰器 | https://shields.io/',
        '- Overleaf - 在线LaTeX | https://www.overleaf.com/register/ (tqin0411@outlook.com)',
        '- DeepL翻译软件 | https://www.deepl.com/translator/',
    ]
    for i in list_3:
        st.markdown(i)





# research **********************************
elif choice == 'Research':
    st.markdown('Research')

    list_3 = [
        '- arXiv每日最新发布论文 | http://arxivdaily.com/',
        '- 文献量化分析可视化VOSViewer | https://www.vosviewer.com/',
        '- 论文图表可视化Origin | https://www.originlab.com/',
        '- LaTex | https://www.latex-project.org/',
        '- 英文论文写作神器(自动语法纠错和书写错误校验，Word中可以安装grammarly插件) | https://www.grammarly.com/',
        '- SCI-Hub | https://www.sci-hub.st/',
        '- SAE (International) Mobilus | https://saemobilus.sae.org/',
        '- Springer | https://link.springer.com/',
        '- ScienceDirect | https://www.sciencedirect.com/',
        '- ResearchGate 科研学术交流平台 | https://www.researchgate.net/',
        '- 中国知网 | https://www.cnki.net/',
        '- arXiv (没经审稿论文的预发表，占坑，免费在线出版。原创性，上传时间戳) | https://arxiv.org/',
        '- 谷歌专利(国外) | http://patents.google.com/',
        '- 佰腾专利查询(国内) | https://www.baiten.cn/',
        '- 中国电子学会考评中心 | https://qceit.org.cn/bos/default.html',
    ]
    for i in list_3:
        st.markdown(i)




# entertainment **********************************
elif choice == 'Entertainment':
    st.markdown('Entertainment')

    list_4 = [
        '- 在线国际象棋 | https://www.chess.com/',
        '- 4K电影下载资源 | https://www.yinfans.me/topic/4k',
        '- 在线影院 | https://www.zbkk.net/'
    ]

    for i in list_4:
        st.markdown(i)
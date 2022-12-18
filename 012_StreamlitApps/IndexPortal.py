###########################
# Create: 2022/12/8
# Author: T
# Purpose: Index apps
###########################

import streamlit as st
import pandas as pd
import numpy as np

st.sidebar.info('IndexPortal [T]')

choice = st.sidebar.selectbox('Category',('Work','Study','Tools','Entertainment'))

# work **********************************
if choice == 'Work':
    st.markdown('Work')





# study **********************************
elif choice == 'Study':
    st.markdown('Study')

    list_2 = [
        '- Pip Tsinghua国内镜像源: https://mirrors.tuna.tsinghua.edu.cn/help/pypi/',
        '- Github: https://github.com/qyu6/ (qyu6, tqin0411@outlook.com)',
        '- Github (homepage): https://github.com/TonyQin0/ (TonyQin0, )',
        '- Github (other): https://github.com/xx/ (tqthooo2021@163.com)',
        '- Databricks社区版: https://community.cloud.databricks.com/login.html/ (tqin0411@outlook.com)', 
        '- Databricks用户社区: https://community.databricks.com/s/login/ (tqin0411@outlook.com)',
        '- Kaggle: https://www.kaggle.com/ (tqin0411@outlook.com)',
        '- Overleaf - 在线LaTeX: https://www.overleaf.com/register/ (tqin0411@outlook.com)',
    ]

    for i in list_2:
        st.markdown(i)







# tools **********************************
elif choice == 'Tools':
    st.markdown('Tools')

    list_3 = [
        '- 在线画思维导图: https://www.iodraw.com/mind',
        '- 115网盘: https://115.com/'

    ]
    for i in list_3:
        st.markdown(i)






# entertainment **********************************
elif choice == 'Entertainment':
    st.markdown('Entertainment')

    list_4 = [
        '- 在线国际象棋: https://www.chess.com/',
        '- 4K电影下载资源: https://www.yinfans.me/topic/4k',
        '- 在线影院: https://www.zbkk.net/'
    ]

    for i in list_4:
        st.markdown(i)
# Databricks notebook source
# MAGIC %md
# MAGIC Agenda
# MAGIC - 展示DBFS下所有文件
# MAGIC - 删除DBFS中的文件/文件夹
# MAGIC - 读取DBFS中的.csv数据到pandas dataframe
# MAGIC - 保存pandas dataframe到DBFS
# MAGIC - 读取DBFS中的.txt数据到pandas dataframe

# COMMAND ----------

# 展示当天DBFS中的所有文件

# path='dbfs:/FileStore/tables/', name='tables/', size=0, modificationTime=0)]
display(dbutils.fs.ls('FileStore/tables/'))

# COMMAND ----------

# 删除DBFS中的文件。返回true-文件删除；返回false-文件不存在
dbutils.fs.rm("/FileStore/tables/Demo_SourceDB.xlsx")

# COMMAND ----------

# 删除DBFS中的文件夹。返回true-文件夹删除；返回false-文件夹不存在
dbutils.fs.rm("/FileStore/tables/test112",True)

# COMMAND ----------

# MAGIC %python
# MAGIC # 读取DBFS中的文件到python dataframe
# MAGIC # <spark-sql>
# MAGIC df = spark.read.csv('/FileStore/tables/gender_submission.csv', header="true", inferSchema="true")
# MAGIC print(df.dtypes)
# MAGIC 
# MAGIC df.select('PassengerId','Survived').show()
# MAGIC 
# MAGIC # 将spark dataframe 转化为 python dataframe，然后在python代码中使用
# MAGIC df1 = df.toPandas()
# MAGIC df1.head(10)

# COMMAND ----------

import pandas as pd

# 将pandas dataframe转为spark dataframe
df2 = spark.createDataFrame(df1)
print(df2.dtypes)
df2.select('PassengerId','Survived').show()

# 将spark dataframe写入DBFS
df2.write.save('/FileStore/tables/test112.csv')

# COMMAND ----------

display(dbutils.fs.ls('/FileStore/tables'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### spark读取DBFS中.txt文件到pandas dataframe，并格式化

# COMMAND ----------

# spark读取DBFS中txt文件
df = spark.read.text('/FileStore/tables/data_multivar.txt')

# txt文件中所有数据会自动到dataframe中的一列
df1 = df.toPandas()

# 将txt中文本格式的数据，通过逗号来拆分开，并保留为string格式
df1['a'], df1['b'], df1['c'], df1['y'] = df1['value'].str.split(',', 3).str
df1.head()

# 删除txt原始的合并列
df1 = df1.drop('value',axis=1)

# 将需要用到的字段，数据格式转化为float
df1[['a','b','c','y']] = df1[['a','b','c','y']].astype('float')

# 查看dataframe数据格式
df1.dtypes

# 提取模型训练的特征X和目标y
X = df1.iloc[:,:-1]
y = df1.iloc[:,-1:]
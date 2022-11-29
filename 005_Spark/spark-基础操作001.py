# Databricks notebook source
# MAGIC %md
# MAGIC spark对数据增删改查

# COMMAND ----------

# 通常项目中的数据不会放在DBFS中，DBFS中的数据对权限管控做不到很精细。同一个databricks工作区的人，数据都是可以查看并访问的
# 要做到精细化管理，可以基于ADLS2(azure data lake storage gen2)来实现，访问时根据文件的绝对路径，比如“abfss://xxxx/xxx/xxx..”
# databricks项目中，表数据可以考虑默认使用delta表的文件类型，在此之前使用.parquet类型更多
# 对于其他类型的数据，存储为文件，但是需要指定格式。通常建模使用的是文件格式存储，调用时对于不同的文件，采取不同的调用方式

# COMMAND ----------

# spark存储不同类型的文件
df.write.saveAsTabel("db_name.table_name") # dataframe保存为表
df.write.parquet("abfss://......") # dataframe保存为parquet文件
df.write.format("delta").save("abfss://......") # dataframe保存为delta
# Databricks notebook source
# MAGIC %md
# MAGIC Pandas练习

# COMMAND ----------

# MAGIC %python
# MAGIC import sys
# MAGIC print(sys.version)
# MAGIC print(sys.executable)
# MAGIC 
# MAGIC import pandas as pd
# MAGIC v = pd.__version__
# MAGIC print('pandas version',v)

# COMMAND ----------

import pandas as pd

df = pd.DataFrame(
    {
        "Name": [
            "Braund, Mr. Owen Harris",
            "Allen, Mr. William Henry",
            "Bonnell, Miss. Elizabeth",
        ],
        "Age": [22, 35, 58],
        "Sex": ["male", "male", "female"],
    }
)

df

# COMMAND ----------

df['Age']

# COMMAND ----------

ages = pd.Series([22,34,38],name='age')
ages

# COMMAND ----------

df['Age'].mean()

# COMMAND ----------

# MAGIC %md
# MAGIC Descriptive statistics

# COMMAND ----------

df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC read & write tabular data

# COMMAND ----------

# MAGIC %python
# MAGIC # [spark-sql]
# MAGIC df = spark.read.csv('/FileStore/tables/gender_submission-2.csv', header="true", inferSchema="true")
# MAGIC print(df.dtypes)
# MAGIC 
# MAGIC df.select('PassengerId','Survived').show()
# MAGIC 
# MAGIC # spark dataframe → python dataframe
# MAGIC df1 = df.toPandas()
# MAGIC df1.head(10)

# COMMAND ----------

dfsource = spark.read.csv('/FileStore/tables/train.csv', header="true", inferSchema="true")
print(dfsource.dtypes)

dfsource.select('PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked').show()

# spark dataframe → python dataframe
df1 = dfsource.toPandas()
df1

# COMMAND ----------

df1.dtypes

# COMMAND ----------

df1.info()

# COMMAND ----------

df1['Name'].head()

print(type(df1['Name']))
print(df1['Name'].shape)

# COMMAND ----------

df1[['Name','Sex','Age']]

print(type(df1[['Name','Sex','Age']]))
print(df1[['Name','Sex','Age']].shape)

# COMMAND ----------

df2 = df1[df1['Age']>35]
df2.head()
df2.shape

# COMMAND ----------

df1['Age']>35

# COMMAND ----------

df = df1
class23 = df[(df['Pclass']==2) | (df['Pclass']==3)]
class23.shape

# COMMAND ----------

age_no_na = df[df['Age'].notna()]
age_no_na.shape

# COMMAND ----------

adult_names = df.loc[df['Age']>35,'Name']
adult_names

# COMMAND ----------

# MAGIC %md
# MAGIC loc和iloc的区别:
# MAGIC - loc可直接用于条件表达式(条件1>a)
# MAGIC - iloc用于矩阵表达式(boolean,[2:3,5:10])

# COMMAND ----------

print(df.head(10))
df.iloc[9:25,2:5].shape

# COMMAND ----------

# MAGIC %md
# MAGIC Create plot in pandas

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

list(df)
dftemp = df[['Age','SibSp','Fare']]
dftemp.plot()
plt.show()

# COMMAND ----------

dftemp['Age'].plot()

# COMMAND ----------

dftemp.plot.scatter(x='Age',y='Fare',alpha=0.5)

# COMMAND ----------

dftemp.plot.box()

# COMMAND ----------

axs = dftemp.plot.line(figsize=(12,4),subplots=True)

# COMMAND ----------

axs = dftemp.plot.area(figsize=(12,4),subplots=True)

# COMMAND ----------

fig,axs = plt.subplots(figsize=(12,4))
dftemp.plot.area(ax=axs)
axs.set_ylabel('dftemp-index')
plt.show()
fig.savefig('test.png')

# COMMAND ----------

# MAGIC %md
# MAGIC create new column

# COMMAND ----------

df['New_Col'] = df['Age']*0.01
df

# COMMAND ----------

list(df)
dftemp = df[['Name','PassengerId','Age']]
dftemp1 = dftemp.rename(columns=
                       {
                           'Name':'col1',
                           'PassengerId':'col2',
                           'Age':'col3'
                       })
dftemp1

# COMMAND ----------

dftemp2 = dftemp.rename(columns=str.lower)
dftemp2

# COMMAND ----------

df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Aggregate - 表聚合

# COMMAND ----------

df[['Sex','Age','Fare']].groupby('Sex').mean()

# COMMAND ----------

df.groupby('Sex').mean(numeric_only=True)

# COMMAND ----------

df.groupby('Sex')['Age'].mean()

# COMMAND ----------

df.groupby(['Sex','Pclass'])['Fare'].mean()

# COMMAND ----------

df['Pclass'].value_counts()

# COMMAND ----------

df.groupby('Pclass')['Pclass'].count()

# COMMAND ----------

# MAGIC %md
# MAGIC Reshape table rows

# COMMAND ----------

df.sort_values(by='Age').head()

# COMMAND ----------

df.sort_values(by=['Pclass','Age'],ascending=False).head()

# COMMAND ----------

dftemp = df.reset_index()
dftemp1 = dftemp.sort_index().groupby(['Pclass']).head(2)
dftemp1

# COMMAND ----------

dftemp1.pivot(columns='Pclass',values='Age')

# COMMAND ----------

dftemp = df.pivot_table(values='Age',index='Sex',columns='Pclass',
                       aggfunc='mean')
dftemp

# COMMAND ----------

# MAGIC %md
# MAGIC - 长数据利用pivot函数，转换为宽数据
# MAGIC - 反过来，宽数据用melt函数，转换为长数据

# COMMAND ----------

# MAGIC %md
# MAGIC Multiple tables can be concatenated both column-wise and row-wise using the concat function.
# MAGIC 
# MAGIC For database-like merging/joining of tables, use the merge function.

# COMMAND ----------

df.Sex.unique()

# COMMAND ----------

# MAGIC %md
# MAGIC Handle textual data

# COMMAND ----------

dftemp = df['Name']
dftemp
dftemp.str.lower()

# COMMAND ----------

dftemp.str.split(',')

# COMMAND ----------

dftemp.str.split(',').str.get(0)

# COMMAND ----------

dftemp.str.contains('Allen')

# COMMAND ----------

df[dftemp.str.contains('Allen')]

# COMMAND ----------

# 计算字符串长度
dftemp.str.len()

# 锁定最长项的ID
dftemp.str.len().idxmax()

# 根据ID定位到字符串内容
df.loc[dftemp.str.len().idxmax(),'Name']

# COMMAND ----------

df['Sex_short'] = df['Sex'].replace({'male':'M',
                                     'female':'F'})
df

# COMMAND ----------

df = pd.DataFrame(
    {"AAA": [4, 5, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
)
df

# COMMAND ----------

df.loc[df.AAA>=5,'BBB']=-1
df

# COMMAND ----------

df.loc[df.AAA<5,['BBB','CCC']]=2000
df

# COMMAND ----------

df_mask = pd.DataFrame(
    {"AAA": [True] * 4, "BBB": [False] * 4, "CCC": [True, False] * 2}
)
df_mask
df.where(df_mask,-1000)

# COMMAND ----------

import numpy as np

df = pd.DataFrame(
    {"AAA": [4, 5, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
)
df
df['logic']=np.where(df['AAA']>5,'high','low')
df

# COMMAND ----------

df[df.AAA<=5]

# COMMAND ----------

df
df.loc[(df["BBB"] < 25) & (df["CCC"] >= -40), "AAA"]

# COMMAND ----------

# argsort() - 按照目标元素从小到大的index进行自动排序，根据的是返回元素对应的index值
df
aValue=43
print((df.CCC - aValue).abs())
df.loc[(df.CCC - aValue).abs().argsort()]

# COMMAND ----------

df = pd.DataFrame(
    {"AAA": [4, 5, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
)
print(df)

crit1 = df.AAA<=5.5
crit2 = df.BBB==10
crit3 = df.CCC> -40

allcrit = crit1 & crit2 & crit3

df[allcrit]


# 条件累加的另一种方式
import functools
critlist = [crit1,crit2,crit3]
allcrit = functools.reduce(lambda x,y: x&y, critlist)
df[allcrit]

# COMMAND ----------

df
df[(df.AAA <= 6) & (df.index.isin([0, 2, 4]))]

# COMMAND ----------

df = pd.DataFrame(
    {"AAA": [4, 5, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]},
    index=["foo", "bar", "boo", "kar"],
)
print(df)

# df.loc['bar':'kar']
df.iloc[0:3]

# COMMAND ----------

# inverse operator(执行相反条件) ~

df = pd.DataFrame(
    {"AAA": [4, 5, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
)

df

df[~((df.AAA <= 6) & (df.index.isin([0, 2, 4])))]

# COMMAND ----------

# applymap() - 根据条件快速映值
df = pd.DataFrame({"AAA": [1, 2, 1, 3], "BBB": [1, 1, 2, 2], "CCC": [2, 1, 3, 1]})

df

source_cols = df.columns
new_cols = [str(x) + "_cat" for x in source_cols]
categories = {1: "Alpha", 2: "Beta", 3: "Charlie"}
df[new_cols] = df[source_cols].applymap(categories.get)
df

# COMMAND ----------

# 根据聚合找到最小值对应索引 - idxmin() to get the index of the minimums
df = pd.DataFrame(
    {"AAA": [1, 1, 1, 2, 2, 2, 3, 3], "BBB": [2, 1, 3, 4, 5, 1, 2, 3]}
)

print(df)
df.loc[df.groupby("AAA")["BBB"].idxmin()]

df.sort_values(by="BBB").groupby("AAA", as_index=False).first()

# COMMAND ----------

# 多索引-multiindexing，堆叠，聚合重组
df = pd.DataFrame(
    {
        "row": [0, 1, 2],
        "One_X": [1.1, 1.1, 1.1],
        "One_Y": [1.2, 1.2, 1.2],
        "Two_X": [1.11, 1.11, 1.11],
        "Two_Y": [1.22, 1.22, 1.22],
    }
)

df

df = df.set_index("row")
df


df.columns = pd.MultiIndex.from_tuples([tuple(c.split("_")) for c in df.columns])
df


# stack & reset. (Try stack(0)和stack(1)的差异，'level-1'是自动添加的)
df = df.stack(0).reset_index(1)
df

df.columns = ['Sample','All_X','All_Y']
df

# COMMAND ----------

# tuple的示例
tuple([1,2,3,4])

tuple({1:2,3:4})

# COMMAND ----------

# Arithmetic - 算数
# Pandas dataframe.div()用于查找数据帧和其他元素的浮点数划分。该函数类似于datafram/other，但提供了额外的支持来处理输入数据之一中的缺失值
cols = pd.MultiIndex.from_tuples(
    [(x, y) for x in ["A", "B", "C"] for y in ["O", "I"]]
)


df = pd.DataFrame(np.random.randn(2, 6), index=["n", "m"], columns=cols)

print(df)

# level:在一个级别上广播，在传递的MultiIndex级别上匹配索引值
df = df.div(df["C"], level=1)

df

# COMMAND ----------

coords = [("AA", "one"), ("AA", "six"), ("BB", "one"), ("BB", "two"), ("BB", "six")]

index = pd.MultiIndex.from_tuples(coords)

df = pd.DataFrame([11, 22, 33, 44, 55], index, ["MyData"])

print(df)

# 切片-Slicing a MultiIndex with xs. 看Level不同带来的差异
df.xs("BB", level=0, axis=0)

df.xs("one", level=1, axis=0)

# COMMAND ----------

# 多重索引切片 - Slicing a MultiIndex with xs, method #2
# itertools
import itertools

index = list(itertools.product(["Ada", "Quinn", "Violet"], ["Comp", "Math", "Sci"]))

headr = list(itertools.product(["Exams", "Labs"], ["I", "II"]))

indx = pd.MultiIndex.from_tuples(index, names=["Student", "Course"])

cols = pd.MultiIndex.from_tuples(headr)  # Notice these are un-named

# %，代表数学运算符号，求模
data = [[70 + x + y + (x * y) % 3 for x in range(4)] for y in range(9)]

df = pd.DataFrame(data, indx, cols)

print(df)

All = slice(None)

df.loc['Violet']
df.loc[(All, "Math"), All]
df.loc[(slice("Ada", "Quinn"), "Math"), All]
df.loc[(All, "Math"), ("Exams")]
df.loc[(All, "Math"), (All, "II")]

# COMMAND ----------

# 排序-sort
print(df)
df.sort_values(by=('Labs','II'),ascending=False)

# COMMAND ----------

# 缺失值-missing data
df = pd.DataFrame(
    np.random.randn(6, 1),
    index=pd.date_range("2013-08-01", periods=6, freq="B"),
    columns=list("A"),
)


df.loc[df.index[3], "A"] = np.nan

print(df)

df.bfill()
df.ffill()

# COMMAND ----------

df = pd.DataFrame(
    {
        "animal": "cat dog cat fish dog cat cat".split(),
        "size": list("SSMMMLL"),
        "weight": [8, 10, 11, 1, 20, 12, 12],
        "adult": [False] * 5 + [True] * 2,
    }
)


print(df)

df.groupby("animal").apply(lambda subf: subf["size"][subf["weight"].idxmax()])


# COMMAND ----------

gb = df.groupby(["animal"])

gb.get_group("cat")

# COMMAND ----------

# Apply to different items in a group
gb = df.groupby(["animal"])
gb
def GrowUp(x):
    avg_weight = sum(x[x["size"] == "S"].weight * 1.5)
    avg_weight += sum(x[x["size"] == "M"].weight * 1.25)
    avg_weight += sum(x[x["size"] == "L"].weight)
    avg_weight /= len(x)
    return pd.Series(["L", avg_weight, True], index=["size", "weight", "adult"])

expected_df = gb.apply(GrowUp)

expected_df

# COMMAND ----------

# Expanding apply - 提供扩展窗口计算(循环累加？)
# functools函数；reduce分解
S = pd.Series([i / 100.0 for i in range(1, 11)])

print(S)

def cum_ret(x, y):
    print('x',x)
    print('y',y)
    return x * (1 + y)

def red(x):
    print('red-x',x)
    return functools.reduce(cum_ret, x, 1.0)

S.expanding().apply(red, raw=True)

# COMMAND ----------

# Replacing some values with mean of the rest of a group
df = pd.DataFrame({"A": [1, 1, 2, 2], "B": [1, -1, 1, 2]})
# print(df)

gb = df.groupby("A")
# print(gb.get_group(1))
# print(df)

def replace(g):
    mask = g < 0
    print('mask=',mask)
    print('g=',g.where(~mask, g[~mask].mean()))
    print('------------------')
    return g.where(~mask, g[~mask].mean())
    

# 当transform作用于整个DataFrame时，实际上就是将传入的所有变换函数作用到每一列中
gb.transform(replace)

# Not fixed. ????

# COMMAND ----------

df = pd.DataFrame(
    {
        "code": ["foo", "bar", "baz"] * 2,
        "data": [0.16, -0.21, 0.33, 0.45, -0.59, 0.62],
        "flag": [False, True] * 3,
    }
)

print(df)

code_groups = df.groupby("code")

# transform() 函数在执行转换后保留与原始数据集相同数量的项目。因此，使用 groupby() 然后使用 transform(sum) 会返回相同的输出
agg_n_sort_order = code_groups[["data"]].transform(sum).sort_values(by="data")

sorted_df = df.loc[agg_n_sort_order.index]
sorted_df

# COMMAND ----------

# Create multiple aggregated columns
rng = pd.date_range(start="2014-10-07", periods=10, freq="2min")

ts = pd.Series(data=list(range(10)), index=rng)

def MyCust(x):
    if len(x) > 2:
        return x[1] * 1.234
    return pd.NaT

# print(ts)

mhc = {"Mean": np.mean, "Max": np.max, "Custom": MyCust}

ts.resample('5min').apply(mhc)

# COMMAND ----------

# Create a value counts column and reassign back to the DataFrame
df = pd.DataFrame(
    {"Color": "Red Red Red Blue".split(), "Value": [100, 150, 50, 50]}
)

df

df["Counts"] = df.groupby(["Color"]).transform(len)

df

# COMMAND ----------

# Shift groups of the values in a column based on the index
df = pd.DataFrame(
    {"line_race": [10, 10, 8, 10, 10, 8], "beyer": [99, 102, 103, 103, 88, 100]},
    index=[
        "Last Gunfighter",
        "Last Gunfighter",
        "Last Gunfighter",
        "Paynter",
        "Paynter",
        "Paynter",
    ],
)


df

# shift为将对应目标值向下移动映射的位数
df["beyer_shifted"] = df.groupby(level=0)["beyer"].shift(2)

df

# COMMAND ----------

# Select row with maximum value from each group
df = pd.DataFrame(
    {
        "host": ["other", "other", "that", "this", "this"],
        "service": ["mail", "web", "mail", "mail", "web"],
        "no": [1, 2, 1, 2, 1],
    }
).set_index(["host", "service"])

print(df)

mask = df.groupby(level=0).agg("idxmax")

df_count = df.loc[mask["no"]].reset_index()

df_count

# COMMAND ----------

# MAGIC %md
# MAGIC # Python user guide 
# MAGIC (from Official docs: https://pandas.pydata.org/docs/user_guide/10min.html )

# COMMAND ----------

import numpy as np
import pandas as pd

dates = pd.date_range("20130101", periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
df

# COMMAND ----------

df.head()
df.tail(3)

df.index

df.columns

df.to_numpy()

# COMMAND ----------

df.describe()

# 转置命令 T-transposing dataframe
df.T

# COMMAND ----------

# axis=0 行索引，axis=1 列索引
df.sort_index(axis=1, ascending=False)

# COMMAND ----------

df.sort_values(by="B")

# COMMAND ----------

# 筛选特定行
df.loc[dates[0]]

# COMMAND ----------

# 减少返回值维度
df.loc["20130102", ["A", "B"]]

# COMMAND ----------

print(df)
df.iat[1,1]

# COMMAND ----------

df[df>0]

# COMMAND ----------

df2 = df.copy()

df2["E"] = ["one", "one", "two", "three", "four", "three"]

df2
df2[df2["E"].isin(["two", "four"])]

# COMMAND ----------

s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("20130102", periods=6))
s1

df['F']=s1
df

df.at[dates[0], "A"] = 0
df.iat[0, 1] = 0
df.loc[:, "D"] = np.array([6] * len(df))
df

# COMMAND ----------

df1 = df.reindex(index=dates[0:5], columns=list(df.columns) + ["E"])

df1.loc[dates[0] : dates[1], "E"] = 1

df1

# drop empty values
df1.dropna(how="any")

# fill empty values
df1.fillna(value=5)

# COMMAND ----------

# 0,1为行列索引的区别
df.mean(1)

# COMMAND ----------

# pandas automatically broadcasts along the specified dimension - 广播机制自动对齐维度
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)

s

df.sub(s, axis="index")

# COMMAND ----------

# np.cumsum - 数组累加
# numpy.cumsum(a, axis=None, dtype=None, out=None)
# axis=0，按照行累加。
# axis=1，按照列累加。
# axis不给定具体值，就把numpy数组当成一个一维数组。

a = np.array([[1,2,3],[4,5,6]])
# print(a)

np.cumsum(a)
np.cumsum(a,axis=1)
np.cumsum(a,axis=0)

print(df)
a=df.apply(np.cumsum)
a

df.apply(lambda x: x.max() - x.min())

# COMMAND ----------

s = pd.Series(np.random.randint(0, 7, size=10))
s.value_counts()

# COMMAND ----------

# 合并-Merging
df = pd.DataFrame(np.random.randn(10, 4),columns=['a','b','c','d'])
df

pieces = [df[:3], df[3:7], df[7:]]
pieces

pd.concat(pieces)

# COMMAND ----------

# join
left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})

right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})

pd.merge(left, right, on="key")

# COMMAND ----------

#☆ stack - 堆叠，将所有列压缩为一列
tuples = list(
    zip(
        ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
        ["one", "two", "one", "two", "one", "two", "one", "two"],
    )
)
tuples
index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])

df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=["A", "B"])
print(df)

df2 = df[:4]
df2

# method “compresses” a level in the DataFrame’s columns
stacked = df2.stack()
stacked.unstack(1)
stacked.unstack(0)

# COMMAND ----------

# pivot table - 数据透视表
df = pd.DataFrame(
    {
        "A": ["one", "one", "two", "three"] * 3,
        "B": ["A", "B", "C"] * 4,
        "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 2,
        "D": np.random.randn(12),
        "E": np.random.randn(12),
    }
)

print(df)

pd.pivot_table(df, values="E", index=["A", "B"], columns=["C"])

# COMMAND ----------

# time series
rng = pd.date_range("1/1/2012", periods=100, freq="S")

ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)

ts.resample("5Min").sum()

ts


rng = pd.date_range("3/6/2012 00:00", periods=5, freq="D")

ts = pd.Series(np.random.randn(len(rng)), rng)

ts


ts_utc = ts.tz_localize("UTC")

ts_utc

# convert time zone
ts_utc.tz_convert("US/Eastern")

# COMMAND ----------

# convert between time span
rng = pd.date_range("1/1/2012", periods=5, freq="M")

ts = pd.Series(np.random.randn(len(rng)), index=rng)

ps = ts.to_period()
ps.to_timestamp()

# COMMAND ----------

prng = pd.period_range("1990Q1", "2000Q4", freq="Q-NOV")
# 按季度划分，且每个年的最后一个月是11月。

ts = pd.Series(np.random.randn(len(prng)), prng)

# asfreq 是一种更改 DatetimeIndex 对象频率的简洁方法
ts.index = (prng.asfreq("M", "e") + 1).asfreq("H", "s") + 9

ts.head()

# COMMAND ----------

df = pd.DataFrame(
    {"id": [1, 2, 3, 4, 5, 6], "raw_grade": ["a", "b", "b", "a", "a", "e"]}
)

df

df["grade"] = df["raw_grade"].astype("category")

df["grade"]

new_categories = ["very good", "good", "very bad"]

df["grade"] = df["grade"].cat.rename_categories(new_categories)
df



# df = pd.DataFrame(
#     {"id": [1, 2, 3, 4, 5, 6], "raw_grade": ["a", "b", "b", "a", "a", "e"]}
# )
print(df)
df["grade"] = df["grade"].cat.set_categories(
    ["very bad", "bad", "medium", "good", "very good"]
)

df["grade"]

# Sorting is per order in the categories, not lexical order:排序按照category的顺序来排
df.sort_values(by="grade")

# COMMAND ----------

import matplotlib.pyplot as plt

plt.close("all")

ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))

# 数组累加-cumsum()函数，时间序列问题中常用
ts1 = ts.cumsum()

# print(ts)
# print(ts1)

ts1.plot()

# COMMAND ----------

df = pd.DataFrame(
    np.random.randn(1000, 4), index=ts.index, columns=["A", "B", "C", "D"]
)

print(df)

df = df.cumsum()

plt.figure();

df.plot();

plt.legend(loc='best')

# COMMAND ----------

np.random.randn(1000, 4)

# COMMAND ----------

# MAGIC %md
# MAGIC Start new session - 20221113
# MAGIC https://pandas.pydata.org/docs/user_guide/dsintro.html

# COMMAND ----------

s = pd.Series(np.random.randn(5), name="something")
s.name

s2=s.rename('different')
s2.name

# COMMAND ----------

data = np.zeros((2,), dtype=[("A", "i4"), ("B", "f4"), ("C", "a10")])

data

# COMMAND ----------

# From a dict of tuples
pd.DataFrame(
    {
        ("a", "b"): {("A", "B"): 1, ("A", "C"): 2},
        ("a", "a"): {("A", "C"): 3, ("A", "B"): 4},
        ("a", "c"): {("A", "B"): 5, ("A", "C"): 6},
        ("b", "a"): {("A", "C"): 7, ("A", "B"): 8},
        ("b", "b"): {("A", "D"): 9, ("A", "B"): 10},
    }
)

# COMMAND ----------

from dataclasses import make_dataclass

Point = make_dataclass("Point", [("x", int), ("y", int)])

pd.DataFrame([Point(0, 0), Point(0, 3), Point(2, 3)])

# COMMAND ----------

# 当我们在写程序时，不确定将来要往函数中传入多少个参数，即可使用可变参数（即不定长参数），用*args,**kwargs表示。
# *args称之为Non-keyword Variable Arguments，无关键字参数；
# **kwargs称之为keyword Variable Arguments，有关键字参数；
# 当函数中以列表或者元组的形式传参时，就要使用*args；
# 当传入字典形式的参数时，就要使用**kwargs

dfa = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

dfa.assign(C=lambda x: x["A"] + x["B"], D=lambda x: x["A"] + x["C"])


# COMMAND ----------

a = 2
a**4

# COMMAND ----------

df1 = pd.DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]}, dtype=bool)
print(df1)
df2 = pd.DataFrame({"a": [0, 1, 1], "b": [1, 1, 0]}, dtype=bool)
print(df2)
df1 & df2

# COMMAND ----------

ser1 = pd.Series([1, 2, 3], index=["a", "b", "c"])

ser2 = pd.Series([1, 3, 5], index=["b", "a", "c"])

print(ser1)

print(ser2)


# numpy.remainder()是另一个用于在numpy中进行数学运算的函数，它返回两个数组arr1和arr2之间的除法元素余数，即 arr1 % arr2 当arr2为0且arr1和arr2都是整数数组时，返回0
# 会按照数组index的对应关系来计算
np.remainder(ser1, ser2)

# COMMAND ----------

pd.set_option("display.width", 40)  # default is 80
pd.set_option("display.max_colwidth", 100)

pd.DataFrame(np.random.randn(3, 12))

# COMMAND ----------

df = pd.DataFrame({"foo1": np.random.randn(5), "foo2": np.random.randn(5)})

df

df.foo1

# COMMAND ----------

# MAGIC %md
# MAGIC essential to basic functions.
# MAGIC https://pandas.pydata.org/docs/user_guide/basics.html

# COMMAND ----------


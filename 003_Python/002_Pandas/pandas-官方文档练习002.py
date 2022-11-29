# Databricks notebook source
# MAGIC %md
# MAGIC https://pandas.pydata.org/docs/user_guide/basics.html

# COMMAND ----------

import pandas as pd
import numpy as np

df = pd.DataFrame(
    {
        "one": pd.Series(np.random.randn(3), index=["a", "b", "c"]),
        "two": pd.Series(np.random.randn(4), index=["a", "b", "c", "d"]),
        "three": pd.Series(np.random.randn(3), index=["b", "c", "d"]),
    }
)
# print(df)

row = df.iloc[1]

column = df["two"]

df.sub(row, axis="columns")

# COMMAND ----------

# Discretization and quantiling | 按区间分段
arr = np.random.randn(20)

factor = pd.cut(arr, 4)

factor


factor = pd.cut(arr, [-5, -1, 0, 1, 5])

factor



# 等量划分
arr = np.random.randn(30)
factor = pd.qcut(arr, [0, 0.25, 0.5, 0.75, 1])
factor
pd.value_counts(factor)

# COMMAND ----------

# pipe - 对于多个function串联使用的场景
import statsmodels.formula.api as sm

df1 = pd.read_csv('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/plyr/baseball.csv')
# df1.to_csv('baseball.csv')
df1

(
    df1.query("h > 0")
    .assign(ln_h=lambda df: np.log(df.h))
    .pipe((sm.ols, "data"), "hr ~ ln_h + year + g + C(lg)")
    .fit()
    .summary()
)

# COMMAND ----------

# aggregation
tsdf = pd.DataFrame(
    np.random.randn(10, 3),
    columns=["A", "B", "C"],
    index=pd.date_range("1/1/2000", periods=10),
)


tsdf.iloc[3:7] = np.nan

tsdf

# tsdf.agg(np.sum)
# tsdf.agg('sum')
# tsdf.sum()

tsdf.agg(["sum", "mean"])

# COMMAND ----------

# Transform with multiple functions, 通过transform一次性传递多个函数
tsdf = pd.DataFrame(
    np.random.randn(10, 3),
    columns=["A", "B", "C"],
    index=pd.date_range("1/1/2000", periods=10),
)

tsdf.iloc[3:7] = np.nan

tsdf

# tsdf.transform(np.abs)
tsdf.transform([np.abs, lambda x: x + 1])

# COMMAND ----------

# Applying elementwise functions
def f(x):
    return len(str(x))

# tsdf['A'].map(f)
tsdf.applymap(f)

# COMMAND ----------

# 定制化字段数据类型，并根据字段类型进行筛选
df = pd.DataFrame(
    {
        "string": list("abc"),
        "int64": list(range(1, 4)),
        "uint8": np.arange(3, 6).astype("u1"),
        "float64": np.arange(4.0, 7.0),
        "bool1": [True, False, True],
        "bool2": [False, True, False],
        "dates": pd.date_range("now", periods=3),
        "category": pd.Series(list("ABC")).astype("category"),
    }
)


df["tdeltas"] = df.dates.diff()

df["uint64"] = np.arange(3, 6).astype("u8")

df["other_dates"] = pd.date_range("20130101", periods=3)

df["tz_aware_dates"] = pd.date_range("20130101", periods=3, tz="US/Eastern")

df

df.dtypes

# COMMAND ----------

# df.select_dtypes(include=[bool])
# df.select_dtypes(include=["bool"])
df.select_dtypes(include=["number", "bool"], exclude=["unsignedinteger"])

# COMMAND ----------

# MAGIC %md
# MAGIC I/O tools
# MAGIC https://pandas.pydata.org/docs/user_guide/io.html

# COMMAND ----------

# 自动将dataframe按照类别，分别存储到不同的子文件夹中去
df = pd.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]})

df.to_parquet(path="test", engine="pyarrow", partition_cols=["a"], compression=None)

# test
# ├── a=0
# │   ├── 0bac803e32dc42ae83fddfd029cbdebc.parquet
# │   └──  ...
# └── a=1
#     ├── e6ab24a4f45147b49b54a662f0c412a3.parquet
#     └── ...

# COMMAND ----------

# 滑动窗口
s = pd.Series(range(5))
print(s)
s.rolling(window=3).sum()

# COMMAND ----------

# rolling windons - central
s = pd.Series(range(10))

s.rolling(window=5).mean()

s.rolling(window=5, center=True).mean()

# COMMAND ----------

# Scaling to large datasets - 处理大数据集的高效方法
# 1.只加载需要的列 2.选择高效的数据存储格式，如parquet 3.优化字段类型，比如一些类别字段，转化为category
import pandas as pd

import numpy as np

def make_timeseries(start="2000-01-01", end="2000-12-31", freq="1D", seed=None):
    index = pd.date_range(start=start, end=end, freq=freq, name="timestamp")
    n = len(index)
    state = np.random.RandomState(seed)
    columns = {
        "name": state.choice(["Alice", "Bob", "Charlie"], size=n),
        "id": state.poisson(1000, size=n),
        "x": state.rand(n) * 2 - 1,
        "y": state.rand(n) * 2 - 1,
    }
    df = pd.DataFrame(columns, index=index, columns=sorted(columns))
    if df.index[-1] == end:
        df = df.iloc[:-1]
    return df


timeseries = [
    make_timeseries(freq="1T", seed=i).rename(columns=lambda x: f"{x}_{i}")
    for i in range(10)
]


ts_wide = pd.concat(timeseries, axis=1)
ts_wide

# COMMAND ----------

ts = ts_wide.copy()
ts

# show memory usage in bytes
ts.memory_usage(deep=True)
# set(ts)

# COMMAND ----------

ts2 = ts.copy()

# ts2["name"] = ts2["name"].astype("category")

ts2.memory_usage(deep=True)



# ts2 = ts.copy()

# ts2["name"] = ts2["name"].astype("category")

# ts2.memory_usage(deep=True)
# Out[18]: 
# Index    8409608
# id       8409608
# name     1051495
# x        8409608
# y        8409608
# dtype: int64

# COMMAND ----------

# Dataset result output in a logical method. Store files in different directories.
import pathlib

N = 12

starts = [f"20{i:>02d}-01-01" for i in range(N)]

ends = [f"20{i:>02d}-12-13" for i in range(N)]

pathlib.Path("data/timeseries").mkdir(exist_ok=True)

for i, (start, end) in enumerate(zip(starts, ends)):
    ts = make_timeseries(start=start, end=end, freq="1T", seed=i)
    ts.to_parquet(f"data/timeseries/ts-{i:0>2d}.parquet")
    
# data
# └── timeseries
#     ├── ts-00.parquet
#     ├── ts-01.parquet
#     ├── ts-02.parquet
#     ├── ts-03.parquet
#     ├── ts-04.parquet
#     ├── ts-05.parquet
#     ├── ts-06.parquet
#     ├── ts-07.parquet
#     ├── ts-08.parquet
#     ├── ts-09.parquet
#     ├── ts-10.parquet
#     └── ts-11.parquet
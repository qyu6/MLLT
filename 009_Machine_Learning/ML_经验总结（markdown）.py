# Databricks notebook source
# MAGIC %md
# MAGIC #### 独热编码(OneHotEncoder)的理解
# MAGIC 离散特征的编码分为两种情况：
# MAGIC - 离散特征的取值之间没有大小的意义，比如color：[red,blue],那么就使用one-hot编码
# MAGIC - 离散特征的取值有大小的意义，比如size:[X,XL,XXL],那么就使用数值的映射{X:1,XL:2,XXL:3}
# MAGIC 
# MAGIC 独热编码即 One-Hot 编码:
# MAGIC - 又称一位有效编码，其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候，其中只有一位有效。可以这样理解，对于每一个特征，如果它有m个可能值，那么经过独热编码后，就变成了m个二元特征（如成绩这个特征有好，中，差变成one-hot就是100, 010, 001）。并且，这些特征互斥，每次只有一个激活。因此，数据会变成稀疏的。
# MAGIC - 将离散特征通过one-hot编码映射到欧式空间，是因为，在回归，分类，聚类等机器学习算法中，特征之间距离的计算或相似度的计算是非常重要的，而我们常用的距离或相似度的计算都是在欧式空间的相似度计算，计算余弦相似性，基于的就是欧式空间
# MAGIC 
# MAGIC one-hot的好处：
# MAGIC - 解决了分类器不好处理属性数据的问题
# MAGIC - 一定程度上起到了扩充特征的作用
# MAGIC 
# MAGIC 实现方式：
# MAGIC - pandas.get_dummies()
# MAGIC - sklearn.preprocessing.OneHotEncoder()
# MAGIC 
# MAGIC 独热编码优缺点：
# MAGIC - 优点：独热编码解决了分类器不好处理属性数据的问题，在一定程度上也起到了扩充特征的作用。它的值只有0和1，不同的类型存储在垂直的空间。
# MAGIC - 缺点：当类别的数量很多时，特征空间会变得非常大。在这种情况下，一般可以用PCA来减少维度。而且one hot encoding+PCA这种组合在实际中也非常有用。
# MAGIC 
# MAGIC 什么情况下(不)用独热编码？
# MAGIC - 用：独热编码用来解决类别型数据的离散值问题
# MAGIC - 不用：将离散型特征进行one-hot编码的作用，是为了让距离计算更合理，但如果特征是离散的，并且不用one-hot编码就可以很合理的计算出距离，那么就没必要进行one-hot编码。有些基于树的算法在处理变量时，并不是基于向量空间度量，数值只是个类别符号，即没有偏序关系，所以不用进行独热编码。 Tree Model不太需要one-hot编码： 对于决策树来说，one-hot的本质是增加树的深度。
# MAGIC 
# MAGIC 什么情况下(不)需要归一化？
# MAGIC - 需要： 基于参数的模型或基于距离的模型，都是要进行特征的归一化。
# MAGIC - 不需要：基于树的方法是不需要进行特征的归一化，例如随机森林，bagging 和 boosting等

# COMMAND ----------

# MAGIC %md
# MAGIC #### sklearn数据预处理中fit(), transform(), fit_transform()差异
# MAGIC 都属于数据预处理的技术
# MAGIC - fit() Method calculates the parameters μ and σ and saves them as internal objects.简单来说，就是求得训练集X的均值啊，方差啊，最大值啊，最小值啊这些训练集X固有的属性。可以理解为一个训练过程
# MAGIC - transform() Method using these calculated parameters apply the transformation to a particular dataset.在Fit的基础上，进行标准化，降维，归一化等操作（看具体用的是哪个工具，如PCA，StandardScaler等）
# MAGIC - fit_transform() joins the fit() and transform() method for transformation of dataset.fit_transform是fit和transform的组合，既包括了训练又包含了转换
# MAGIC 
# MAGIC fit_transform(trainData)对部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等。根据对之前部分trainData进行fit的整体指标，对剩余的数据（testData）使用同样的均值、方差、最大最小值等指标进行转换transform(testData)，从而保证train、test处理方式相同。所以，一般都是这么用：
# MAGIC 
# MAGIC - from sklearn.preprocessing import StandardScaler
# MAGIC - sc = StandardScaler()
# MAGIC - sc.fit_tranform(X_train)
# MAGIC - sc.tranform(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 数组(n,)和(n,1)的区别？
# MAGIC - (n,) 是一维数组，一个中括号
# MAGIC - (n,1) 是二维数组，两个中括号
# MAGIC - ravel( ), 将多维数组变为一维数组
# MAGIC - reshape( ), 改变数组的维度

# COMMAND ----------

# MAGIC %md
# MAGIC #### 获取公开数据集的地址
# MAGIC - https://archive.ics.uci.edu/ml/index.php

# COMMAND ----------

# MAGIC %md
# MAGIC #### 决策树能做回归吗？
# MAGIC 能，回归就是根据特征向量来决定对应的输出值。回归树就是将特征空间划分成若干单元，每一个划分单元有一个特定的输出。对于测试数据，只要按照特征将其归到某个单元，便得到对应的输出值。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 逻辑回归只能用于二分类吗？
# MAGIC - 原理上，是。逻辑回归是基于概率论的，原理上只能解决二分类问题
# MAGIC - 应用上，不是。实际应用上，采取OVR(One .vs Rest)的思想，一类是A类，另外是其他；以此方式重复应用在其他类别。

# COMMAND ----------

# MAGIC %md
# MAGIC #### enumerate()函数含义
# MAGIC - 枚举遍历函数
# MAGIC - enumerate(iteration, start)函数默认包含两个参数，其中iteration参数为需要遍历的参数，比如字典、列表、元组等，start参数为开始的参数，默认为0（不写start那就是从0开始）。enumerate函数有两个返回值，第一个返回值为从start参数开始的数，第二个参数为iteration参数中的值

# COMMAND ----------

# MAGIC %md
# MAGIC #### 算法的时间复杂度和空间复杂度的区别？
# MAGIC - 时间复杂度：需要多少时间才能求解
# MAGIC - 空间复杂度：需要多少内存才能满足求解需求
# MAGIC - 一个好的算法往往更注重的是时间复杂度的比较，而空间复杂度只要在一个合理的范围内就可以

# COMMAND ----------

# MAGIC %md
# MAGIC #### numpy.empty()
# MAGIC 按照指定数组维度，生成随机数组
# MAGIC 
# MAGIC 用例:
# MAGIC numpy.empty(shape, dtype=float, order=‘C’)
# MAGIC 
# MAGIC 功能:
# MAGIC 根据给定的维度和数值类型返回一个新的数组，其元素不进行初始化。
# MAGIC 
# MAGIC 参数
# MAGIC 
# MAGIC 变量名	数据类型	功能
# MAGIC shape	整数或者整数组成的元组	空数组的维度，例如：(2, 3)或者2
# MAGIC dtype	数值类型，可选参数	指定输出数组的数值类型，例如numpy.int8。默认为numpy.float64。
# MAGIC order	{‘C’, ‘F’}，可选参数	是否在内存中以C或fortran(行或列)顺序存储多维数据

# COMMAND ----------

# MAGIC %md
# MAGIC #### 工业4.0典型场景 - 模型字典框架
# MAGIC 
# MAGIC ###### 模型训练及发布环节
# MAGIC 
# MAGIC |字典条目属性|内容|备注|
# MAGIC |---|---|---|
# MAGIC |模型名称|外观脏污模型|训练模型时用户自定义设置输出模型的名称，可根据业务场景、管理模式自行定义规范|
# MAGIC |训练人员|xx|执行模型训练任务，生成此模型文件的平台用户|
# MAGIC |创建时间|xx|模型对应的训练任务启动运行的时间|
# MAGIC |发布名称|外观脏污模型|训练模型时用户自定义设置输出模型的名称，可根据业务场景、管理模式自行定义规范|
# MAGIC |模型训练人员|axx|执行模型训练任务，生成此模型文件的平台用户|
# MAGIC |模型发布人员|bxx|发布模型资源的平台用户|
# MAGIC |任务类型|缺陷检测|模型解决的是什么类型的任务，比如缺陷检测、零件识别|
# MAGIC |训练数据|外观图片xx|训练此模型所使用的数据资源|
# MAGIC |模型框架|pytorch 1.8|一般指深度学习的模型开发框架，如TensorFlow，pytorch等，也可能指在其上进一步封装的框架，如detectron等|
# MAGIC |测试精度|MAP:67.2等|模型性能精度的简略指标，在经过模型测试后可给出一个参考指标|
# MAGIC |主干网络|Resnet101|CNN模型的主干网络，Alexnet，Densnet， Resnet等|
# MAGIC |检测/分割模型算法|faster_RCNN|复杂图像任务的算法类型或算法架构，层级介于框架和主干网络之间|
# MAGIC |适用范围|xx工位L4机台|模型适用的机台场景|
# MAGIC |所属产品|/|模型用于哪个产品、应用进行部署|
# MAGIC 
# MAGIC 
# MAGIC ###### 模型快速部署环节
# MAGIC |字典条目属性|内容|备注|
# MAGIC |---|---|---|
# MAGIC |发布名称|外观脏污模型|训练模型时用户自定义设置输出模型的名称，可根据业务场景、管理模式自行定义规范|
# MAGIC |发布人员|xx|发布模型资源的平台用户|
# MAGIC |任务类型|缺陷检测|模型解决的是什么类型的任务，比如缺陷检测、零件识别等|
# MAGIC |模型检测(或识别)的类别(特征、缺陷)|1.脏污 2.划伤|此模型可识别或检测哪些类别|
# MAGIC |训练数据|外观图片xx|训练此模型所使用的数据资源|
# MAGIC |模型框架|pytorch 1.8|一般指深度学习的模型开发框架，如TensorFlow，pytorch等，也可能指在其上进一步封装的框架，如detectron等|
# MAGIC |测试精度|MAP:67.2等|模型性能精度的简略指标，在经过模型测试后可给出一个参考指标|
# MAGIC |主干网络|Resnet101|CNN模型的主干网络，Alexnet，Densnet， Resnet等|
# MAGIC |检测/分割模型算法|faster_RCNN|复杂图像任务的算法类型或算法架构，层级介于框架和主干网络之间|
# MAGIC |适用范围|xx工位L4机台|模型适用的机台场景|
# MAGIC |所属产品|/|模型用于哪个产品、应用进行部署|

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### <待更新问题>
# MAGIC - 傅里叶变换，从傅里叶级数到快速傅里叶变换（FTT）的关系
# MAGIC - 高斯核，调和分析。用核函数提取特征，也可以用其他核函数。高斯核相对好算，傅里叶变换也用的是高斯定理，这种方法在卷积定理中经常被使用
# MAGIC - 卷积层，池化层，全连接层的含义和作用
# MAGIC - 平衡模型的鲁棒性和精确性
# MAGIC - LSTM - RNN(时间序列预测)

# COMMAND ----------


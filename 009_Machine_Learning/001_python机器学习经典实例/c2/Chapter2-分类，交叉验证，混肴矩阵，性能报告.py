# Databricks notebook source
# MAGIC %md
# MAGIC Agenda:
# MAGIC - 简单分类器
# MAGIC - 逻辑回归分类，logistics regression classifer
# MAGIC - 朴素贝叶斯分类, naive bayes classifier
# MAGIC - 交叉验证, cross validation
# MAGIC - 混肴矩阵, confusion matrix
# MAGIC - 模型性能报告, model performance
# MAGIC - Random Forest Classifier - 根据汽车特征评估质量
# MAGIC - 模型验证曲线, validation curve (超参数遍历+交叉验证 = DOE矩阵计算结果)
# MAGIC - 学习曲线，learning curve

# COMMAND ----------

# MAGIC %md
# MAGIC #### 简单分类器

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

# input data
X = np.array([[3,1], [2,5], [1,8], [6,4], [5,2], [3,5], [4,7], [4,-1]])

# labels
y = [0, 1, 1, 0, 0, 1, 1, 0]

# separate the data into classes based on 'y'
class_0 = np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i]==1])

# plot input data
plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], color='black', marker='s')
plt.scatter(class_1[:,0], class_1[:,1], color='black', marker='x')

# draw the separator line
line_x = range(10)
line_y = line_x

# plot labeled data and separator line 
# plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], color='black', marker='s')
plt.scatter(class_1[:,0], class_1[:,1], color='black', marker='x')
plt.plot(line_x, line_y, color='black', linewidth=3)

# plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 逻辑回归分类-logistic regression classifer
# MAGIC - 对给定的数据点，建立一个可以在类之间绘制线性边界的模型
# MAGIC - 对训练数据派生的一组方程进行求解来提取边界
# MAGIC - LR原理上只用于二分类；实际应用过程采取OVR(one .vs rest)的思想，应用于多分类问题

# COMMAND ----------

import numpy as np
from sklearn import linear_model 
import matplotlib.pyplot as plt

# 创建一些带训练标记的样本数据,三个类别
X = np.array([[4, 7], [3.5, 8], [3.1, 6.2], [0.5, 1], [1, 2], [1.2, 1.9], [6, 2], [5.7, 1.5], [5.4, 2.2]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

# initialize the logistic regression classifier
# solver - 设置求解系统方程的算法类型
# C - 表示正则化的强度，数值越大，正则化的强度越高，分类错误的惩罚值越高。C的值越大，每个类别的分类边界更优
classifier = linear_model.LogisticRegression(solver='liblinear', C=100000)

# train the classifier
classifier.fit(X, y)

# 可视化分类结果
def plot_classifier(classifier, X, y):
    # define ranges to plot the figure - 设置可视化边界+1
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0

    # denotes the step size that will be used in the mesh grid - 设置可视化图片像素颗粒度
    step_size = 0.1

    # define the mesh grid
    # 为了画出边界，需要利用一组网格(grid)数据求出方程的值，然后把分类边界画出来
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # compute the classifier output
    # np.c_ 中的c 是 column(列)的缩写，就是按列叠加两个矩阵，就是把两个矩阵左右组合
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

    # reshape the array
    mesh_output = mesh_output.reshape(x_values.shape)

    # Plot the output using a colored plot 
    plt.figure()
#     print(mesh_output)
    
    # choose a color scheme you can find all the options 
    # here: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.tab10)
    
    print(y)
    # Overlay the training points on the plot 
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='white', linewidth=2, cmap=plt.cm.tab10)

    # specify the boundaries of the figure
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

    # specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(min(X[:, 0])-1), int(max(X[:, 0])+1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1])-1), int(max(X[:, 1])+1), 1.0)))

#     plt.show()


# draw datapoints and boundaries
plot_classifier(classifier, X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 朴素贝叶斯分类器-Naive Bayes
# MAGIC - 贝叶斯定理进行建模的监督学习分类器
# MAGIC - 朴素贝叶斯算法（Naive Bayesian) 是应用最为广泛的分类算法之一
# MAGIC - 朴素贝叶斯方法是在贝叶斯算法的基础上进行了相应的简化，即假定给定目标值时属性之间相互条件独立。也就是说没有哪个属性变量对于决策结果来说占有着较大的比重，也没有哪个属性变量对于决策结果占有着较小的比重。虽然这个简化方式在一定程度上降低了贝叶斯分类算法的分类效果，但是在实际的应用场景中，极大地简化了贝叶斯方法的复杂性
# MAGIC - 朴素贝叶斯分类（NBC）是以贝叶斯定理为基础并且假设特征条件之间相互独立的方法，先通过已给定的训练集，以特征词之间独立作为前提假设，学习从输入到输出的联合概率分布，再基于学习到的模型，输入X求出使得后验概率最大的输出y

# COMMAND ----------

# spark读取DBFS中txt文件
df = spark.read.text('/FileStore/tables/data_multivar-1.txt')

# txt文件中所有数据会自动到dataframe中的一列
df1 = df.toPandas()

# 将txt中文本格式的数据，通过逗号来拆分开，并保留为string格式
df1['a'], df1['b'], df1['y'] = df1['value'].str.split(',', 3).str
df1.head()

# 删除txt原始的合并列
df1 = df1.drop('value',axis=1)

# 将需要用到模型训练的字段，数据格式转化为float
df1[['a','b','y']] = df1[['a','b','y']].astype('float')
df1[['y']] = df1[['y']].astype('int')

# 查看dataframe数据格式
print(df1.dtypes)

df1

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# COMMAND ----------

X=np.array(df1.iloc[:,:-1])
y=np.array(df1.iloc[:,-1])

classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X, y)
y_pred = classifier_gaussiannb.predict(X)

# compute accuracy of the classifier
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of the classifier =", round(accuracy, 2), "%")


y.shape
type(X[:,0])

# COMMAND ----------

# 可视化分类结果
def plot_classifier(classifier, X, y):
    # define ranges to plot the figure - 设置可视化边界+1
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0

    # denotes the step size that will be used in the mesh grid - 设置可视化图片像素颗粒度
    step_size = 0.1

    # define the mesh grid
    # 为了画出边界，需要利用一组网格(grid)数据求出方程的值，然后把分类边界画出来
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # compute the classifier output
    # np.c_ 中的c 是 column(列)的缩写，就是按列叠加两个矩阵，就是把两个矩阵左右组合
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

    # reshape the array
    mesh_output = mesh_output.reshape(x_values.shape)

    # Plot the output using a colored plot 
    plt.figure()
    
#     print(mesh_output)
    # choose a color scheme you can find all the options 
    # here: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.tab20c)
    
#     print(y)
    # Overlay the training points on the plot 
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='white', linewidth=1.5, cmap=plt.cm.tab20c)

    # specify the boundaries of the figure
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

    # specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(min(X[:, 0])-1), int(max(X[:, 0])+1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1])-1), int(max(X[:, 1])+1), 1.0)))

#     plt.show()


# draw datapoints and boundaries
plot_classifier(classifier_gaussiannb, X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 训练集和测试集自动拆分

# COMMAND ----------

# Train test split (在sklearn = 0.18版本之后，cross_validation被废弃，取而代之的是model_selection)
from sklearn import model_selection

# 25%-测试数据，75%-训练数据
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
classifier_gaussiannb_new = GaussianNB()
classifier_gaussiannb_new.fit(X_train, y_train)
y_test_pred = classifier_gaussiannb_new.predict(X_test)

# compute accuracy of the classifier
accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of the classifier =", round(accuracy, 2), "%")

plot_classifier(classifier_gaussiannb_new, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 交叉验证 - Cross Validation
# MAGIC - 为了让模型更稳定，需要用数据集的不同自己来反复验证模型性能
# MAGIC - 模型性能评估常用指标：精度-precision，召回率-recall，F1 score
# MAGIC - F1 score是精度和召回率的调和均值(harmonic mean). F1 score = 2 * precision * recall / (precision + recall)
# MAGIC - k折交叉验证的步骤：
# MAGIC   - 将数据集均分成k份 
# MAGIC   - 不重复地每次取一份作为测试集，其余做训练集，之后分别计算测试集的MSE等model metrics 
# MAGIC   - 将k次的MSE取平均得到最终的MSE值

# COMMAND ----------

# Cross validation and scoring functions

num_validations = 5 # cv_number的选取是一个Bias和Variance的trade-off，一般选5或10

# 常用metrics scoring计算方法
accuracy = model_selection.cross_val_score(classifier_gaussiannb, X, y, scoring='accuracy', cv=num_validations)
f1 = model_selection.cross_val_score(classifier_gaussiannb, X, y, scoring='f1_weighted', cv=num_validations)
precision = model_selection.cross_val_score(classifier_gaussiannb, X, y, scoring='precision_weighted', cv=num_validations)
recall = model_selection.cross_val_score(classifier_gaussiannb, X, y, scoring='recall_weighted', cv=num_validations)

# 输出结果
print('\n**** Classifier_gaussiannb: ****\n')
print("Accuracy: " + str(round(100*accuracy.mean(), 2)) + "%")
print("F1: " + str(round(100*f1.mean(), 2)) + "%")
print("Precision: " + str(round(100*precision.mean(), 2)) + "%")
print("Recall: " + str(round(100*recall.mean(), 2)) + "%")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 混淆矩阵-Confusion Matrix
# MAGIC - 是理解分类模型性能的数据表，有助于帮助理解模型如何把数据分成不同的类
# MAGIC - 了解数据被错误分类的情况，便于找到模型优化的切入点

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Show confusion matrix
def plot_confusion_matrix(confusion_mat):
    plt.imshow(confusion_mat, interpolation='nearest', origin='lower', cmap=plt.cm.Greens)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

y_true = [1, 0, 0, 2, 1, 0, 3, 3, 3]
y_pred = [1, 1, 0, 2, 1, 0, 1, 3, 3]
confusion_mat = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(confusion_mat) # 从结果看，混淆矩阵对角线的颜色越深，分类的效果越好

# COMMAND ----------

# Print classification report - 直接输出模型性能metrics
from sklearn.metrics import classification_report
target_names = ['Class-0', 'Class-1', 'Class-2', 'Class-3']
print(classification_report(y_true, y_pred, target_names=target_names)) 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 随机森林分类 - Random Forest Classifier
# MAGIC - 实例：根据汽车特征评估汽车质量（根据车门数量，后备箱大小，维修成本等来确定汽车质量。最后分为不达标，达标，良好，优秀四个评级）
# MAGIC - 随机森林既可以做回归，又可以做分类

# COMMAND ----------

import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# COMMAND ----------

# spark读取DBFS中txt文件
df = spark.read.text('/FileStore/tables/car_data.txt')

# txt文件中所有数据会自动到dataframe中的一列
df1 = df.toPandas()

# 将txt中文本格式的数据，通过逗号来拆分开，并保留为string格式
df1['buying'], df1['maint'], df1['doors'], df1['persons'], df1['lug_boot'], df1['safety'], df1['quality'] = df1['value'].str.split(',', 6).str
df1.head()

# 删除txt原始的合并列
df1 = df1.drop('value',axis=1)

# # 将需要用到模型训练的字段，数据格式转化为float
# df1[['a','b','y']] = df1[['a','b','y']].astype('float')
# df1[['y']] = df1[['y']].astype('int')

# 查看dataframe数据格式
print(df1.dtypes)

df1

# COMMAND ----------

X = np.array(df1)
X

# Convert string data to numerical data
label_encoder = [] 

# empty(shape[, dtype, order]) 依给定的shape, 和数据类型 dtype,  返回一个一维或者多维数组，数组的元素不为空，为随机产生的数据。
X_encoded = np.empty(X.shape)


# enumerate - 枚举遍历
for i,item in enumerate(X[0]):
#     print(X[0])
#     print(i)
#     print(item)
#     print(X[:,i].shape)
    
    label_encoder.append(preprocessing.LabelEncoder())
#     print(X[:,i])
#     print('---')
    X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
    
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)


# Build a Random Forest classifier
params = {'n_estimators': 200, 'max_depth': 8, 'random_state': 7}
classifier = RandomForestClassifier(**params)
classifier.fit(X, y)

# Cross validation - 交叉验证
from sklearn import model_selection

accuracy = model_selection.cross_val_score(classifier, X, y, scoring='accuracy', cv=3)
print("Accuracy of the classifier: " + str(round(100*accuracy.mean(), 2)) + "%")

# # **********************************************************************
# # Testing encoding on single data instance
# input_data = ['vhigh', 'vhigh', '2', '2', 'small', 'low']
# input_data_encoded = [-1] * len(input_data)

# for i,item in enumerate(input_data):
# #     input_data_encoded[i] = int(label_encoder[i].transform(input_data[i]))
#     print(input_data)
#     print(i)
# #     print(item)
# #     print(input_data[i].shape)
#     input_data[i] = np.array(input_data[i]).reshape(-1)
#     print(input_data[i])
#     input_data_encoded[i] = label_encoder[-1].transform(input_data[i])

# input_data_encoded = np.array(input_data_encoded)

# # Predict and print output for a particular datapoint
# output_class = classifier.predict(input_data_encoded)
# print("Output class:", label_encoder[-1].inverse_transform(output_class)[0])
# # **********************************************************************

# COMMAND ----------

# MAGIC %md
# MAGIC #### 验证曲线 - Validation Curve

# COMMAND ----------

# 生成验证曲线 - validation curve. 
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html
# 通过调整两个主要超参数(hyperparameters): n_estimators, max_depth来体现模型的性能，帮助理解每个超参数对训练得分的影响

# 固定max_depth值，对n_estimators参数进行验证：
from sklearn.model_selection import validation_curve

classifier = RandomForestClassifier(max_depth=4, random_state=7)

# 创建分段的等差序列，定义超参数的分布区间
parameter_range = np.linspace(5, 200, 20).astype(int)

train_scores, validation_scores = validation_curve(classifier, X, y, "n_estimators", parameter_range, cv=5)
print("\n***** VALIDATION CURVES *****")
# 返回数组结果 → arrayz_of_shape(n_ticks, n_cv_folds)
print("\nParam: n_estimators\nTraining scores:\n", train_scores)
print("\nParam: n_estimators\nValidation scores:\n", validation_scores)

# Plot the curve
plt.figure()
plt.plot(parameter_range, 100*np.average(train_scores, axis=1), color='black') # np.average(..axis=1) - 矩阵的行均值
plt.title('Training curve')
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.show()

# COMMAND ----------

# 固定n_estimators值，对max_depth参数进行验证： 
classifier = RandomForestClassifier(n_estimators=20, random_state=7)
parameter_range = np.linspace(2, 15, 10).astype(int)

train_scores, valid_scores = validation_curve(classifier, X, y,"max_depth", parameter_range, cv=5)
print("\n***** VALIDATION CURVES *****")
print("\nParam: max_depth\nTraining scores:\n", train_scores)
print("\nParam: max_depth\nValidation scores:\n", validation_scores)

# Plot the curve
plt.figure()
plt.plot(parameter_range, 100*np.average(train_scores, axis=1), color='black')
plt.title('Validation curve')
plt.xlabel('Maximum depth of the tree')
plt.ylabel('Accuracy')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 学习曲线 - Learning Curve

# COMMAND ----------

# Learning curves
# 学习曲线帮助理解数据集大小对机器学习模型的影响。尤其是算力有限时，需要平衡考虑。学习曲线就是通过画出不同训练集大小时训练集和交叉验证的准确率，可以看到模型在新数据上的表现，进而来判断模型是否方差偏高或偏差过高，以及增大训练集是否可以减小过拟合。

from sklearn.model_selection import learning_curve

classifier = RandomForestClassifier()
# classifier = RandomForestClassifier(random_state=7)

# parameter_grid = np.array([50, 100, 200, 500, 800, 1100, 1300])
parameter_grid = np.linspace(1, 0.78*len(X), 20).astype(int)
train_sizes, train_scores, validation_scores = learning_curve(classifier, X, y, train_sizes=parameter_grid, cv=5)
print("\n***** LEARNING CURVES *****")
print("\nTraining sizes:\n", train_sizes)
print("\nTraining scores:\n", train_scores)
print("\nValidation scores:\n", validation_scores)

# Plot the curve
plt.figure()
plt.plot(parameter_grid, 100*np.average(train_scores, axis=1), color='red', marker='o')
plt.plot(parameter_grid, 100*np.average(validation_scores, axis=1), color='green', marker='o')

plt.legend(["train_scores", "validation_scores"], loc ="upper left") 
plt.grid()
plt.title('Learning curve')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 实例：估算收入阶层
# MAGIC - Naive Bayes，GaussianNB

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

# COMMAND ----------

# spark读取DBFS中txt文件
df = spark.read.text('/FileStore/tables/adult_data.txt')

# txt文件中所有数据会自动到dataframe中的一列
df1 = df.toPandas()

# 将txt中文本格式的数据，通过逗号来拆分开，并保留为string格式
expand_df = df1['value'].str.split(',', expand=True)
expand_df

# 暴力字典法 - 批量遍历修改数据集的列名
new_dict = {key:'Column'+'_'+str(i) for i, key in enumerate(expand_df.columns)}
expand_df.rename(columns=new_dict, inplace=True)

expand_df

# COMMAND ----------

# 平衡样本类别数量，保证初始类型没有偏差。两种类别各取10000个样本点
# Reading the data
X = []
y = []
count_lessthan50k = 0
count_morethan50k = 0
num_images_threshold = 10000

for i in range(len(expand_df)):
    if expand_df.iloc[0,]

test1 = expand_df.iloc[1,:-1]
test2 = expand_df.iloc[2,:-1]
X.append(test1)
X.append(test2)

b=np.array(X)
len(expand_df)

# expand_df

# COMMAND ----------


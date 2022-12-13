import neurolab as nl
import numpy as np
import matplotlib.pyplot as plt

# Generate training data
min_value = -12
max_value = 12
num_datapoints = 90

# 训练数据由我们定义的函数组成。神经网络将根据输入和输出数据来学习
x = np.linspace(min_value, max_value, num_datapoints)
y = 2 * np.square(x) + 7

# linalg = linear(线性) + algebra(代数)，norm表示范数
# npllinalg.norm默认参数 - 矩阵整体元素平方和开根号，不保留矩阵二维特征
# https://blog.csdn.net/hqh131360239/article/details/79061535
y /= np.linalg.norm(y)

# 数组变形
data = x.reshape(num_datapoints, 1)
labels = y.reshape(num_datapoints, 1)

# Plot input data
plt.figure()
plt.scatter(data, labels)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Input data')

# Define a multilayer neural network with 2 hidden layers;
# Each hidden layer consists of 10 neurons and the output layer 
# consists of 1 neuron
# 定义一个深度神经网络，包含两个隐藏层，每个隐藏层由10个神经元组成，输出层由一个神经元组成
multilayer_net = nl.net.newff([[min_value, max_value]], [10, 10, 1])

# Change the training algorithm to gradient descent
# 设置训练算法为梯度下降法：https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/
multilayer_net.trainf = nl.train.train_gd

# Train the network-训练网络
error = multilayer_net.train(data, labels, epochs=800, show=100, goal=0.01)

# Predict the output for the training inputs 
# 用训练数据运行网络，预测结果，并查看性能表现
predicted_output = multilayer_net.sim(data)

# Plot training error-画出训练误差
plt.figure()
plt.plot(error)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')

# Plot prediction-画出预测结果。输入一组新的输入数据，运行神经网络，看性能表现
x2 = np.linspace(min_value, max_value, num_datapoints * 2)
y2 = multilayer_net.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3 = predicted_output.reshape(num_datapoints)

plt.figure()
# 原始数据y，训练数据结果-y3，测试数据结果y2
# plt.plot(x2, y2, '-', x, y, '.', x, y3, '*')

plt.plot(x,y,marker = '^',linestyle='-',alpha=0.7,color='blue')
plt.plot(x2,y2,marker = '*',linestyle='-',alpha=0.7,color='orange')
plt.plot(x,y3,marker = 'x',linestyle='-',alpha=0.7,color='purple')
plt.legend(('Source data','Test data','Train data'),loc='lower left')
plt.title('Ground truth vs predicted output')
# plt.legend()

plt.show()
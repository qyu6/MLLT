import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# Define input data
input_file = 'data_single_layer.txt'
input_text = np.loadtxt(input_file)
data = input_text[:, 0:2]
labels = input_text[:, 2:]

# Plot input data
plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Input data')

# Min and max values for each dimension
# 提取每个维度的最小值和最大值，作为输入
x_min, x_max = data[:,0].min(), data[:,0].max()
y_min, y_max = data[:,1].min(), data[:,1].max()

# Define a single-layer neural network with 2 neurons;
# Each element in the list (first argument) specifies the 
# min and max values of the inputs
# 定义一个单层神经网络，隐藏层包含两个神经元。
# 在感知机第一个参数的每个元素中指定参数的最大值和最小值
single_layer_net = nl.net.newp([[x_min, x_max], [y_min, y_max]], 2)

# Train the neural network
# 通过50次迭代训练该神经网络
error = single_layer_net.train(data, labels, epochs=50, show=2000, lr=0.01)

# Plot results
plt.figure()
plt.plot(error)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.title('Training error progress')
plt.grid()

plt.show()

print ('[0.3,4.5] predict result →',single_layer_net.sim([[0.3, 4.5]]))
print ('[4.5,0.5] predict result →',single_layer_net.sim([[4.5, 0.5]]))
print ('[4.3,8] predict result →',single_layer_net.sim([[4.3, 8]]))
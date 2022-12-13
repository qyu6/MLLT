import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

# Define input data
data = np.array([[0.3, 0.2], [0.1, 0.4], [0.4, 0.6], [0.9, 0.5]])
labels = np.array([[0], [0], [0], [1]])

# Plot input data
plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Input data')

# Define a perceptron with 2 inputs;
# Each element of the list in the first argument 
# specifies the min and max values of the inputs
# 定义一个感知器perceptron，有两个输入。函数还需要限定输入数据的最大值和最小值
# 定义由两个输入的感知器，在感知器第一个参数的每个元素中指定参数的最大值和最小值
perceptron = nl.net.newp([[0, 1],[0, 1]], 1)

# Train the perceptron-训练感知器
# epochs - 指定了训练数据集需要完成的测试次数
# show - 指定了显示训练过程的频率
# lr - 制定了感知器的学习速度。学习率是指学习算法在参数空间中搜索的补偿。如果太大，算法行进的很快，但可能错失最优值。如果太小，训练很慢，但可以达到最优值。
error = perceptron.train(data, labels, epochs=50, show=15, lr=0.01)

# plot results
plt.figure()
plt.plot(error)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.grid()
plt.title('Training error progress')

plt.show()
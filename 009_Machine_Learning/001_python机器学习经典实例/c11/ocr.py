import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

# Input file
input_file = 'letter.data'

# Number of datapoints to load from the input file
# 神经网络处理大量数据时，会花费很多时间。此时只用20个数据点
num_datapoints = 20

# 前20行数据有7个不同的字符
# Distinct characters-不同的字符
orig_labels = 'omandig'

# Number of distinct characters-不同字符的数量
num_output = len(orig_labels)

# Training and testing parameters
# 90%训练集，10%测试集
num_train = int(0.9 * num_datapoints)
num_test = num_datapoints - num_train

# Define dataset extraction parameters
# 定义数据集提取参数，数据文件中的每行起始索引值和终止索引值 
start_index = 6
end_index = -1

# Creating the dataset-生成数据集
data = []
labels = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        # Split the line tabwise-用tab键分割
        list_vals = line.split('\t')

        # If the label is not in our ground truth labels, skip it
        # 增加一个错误检查步骤，检查这些字符是否在标签中
        if list_vals[1] not in orig_labels:
            continue

        # Extract the label and append it to the main list
        # 提取标签，将其添加到主列表后面
        label = np.zeros((num_output, 1))
        label[orig_labels.index(list_vals[1])] = 1
        labels.append(label)

        # Extract the character vector and append it to the main list
        # 提取字符，将其添加到主列表后面
        cur_char = np.array([float(x) for x in list_vals[start_index:end_index]])
        data.append(cur_char)

        # Exit the loop once the required dataset has been loaded
        # 有足够多数据时跳出循环
        if len(data) >= num_datapoints:
            break

# Convert data and labels to numpy arrays-转成np数组
data = np.asfarray(data)
labels = np.array(labels).reshape(num_datapoints, num_output)

# Extract number of dimensions-提取数组维度信息
num_dims = len(data[0])

# Create and train neural network-训练神经网络
net = nl.net.newff([[0, 1] for _ in range(len(data[0]))], [128, 16, num_output])
net.trainf = nl.train.train_gd
error = net.train(data[:num_train,:], labels[:num_train,:], epochs=20000, 
        show=100, goal=0.005)

plt.plot(error)
plt.xlabel('Number of epochs')
plt.ylabel('Error (MSE)')

# Predict the output for test inputs - 为测试输入数据预测输出结构
predicted_output = net.sim(data[num_train:, :])
print ("\nTesting on unknown data:")
for i in range(num_test):
    print ("\nOriginal:", orig_labels[np.argmax(labels[i])])
    print ("Predicted:", orig_labels[np.argmax(predicted_output[i])])


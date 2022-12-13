import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# 定义函数，函数利用输入参数创建一个波形
def create_waveform(num_points):
    # Create train samples
    data1 = 1 * np.cos(np.arange(0, num_points))
    data2 = 2 * np.cos(np.arange(0, num_points))
    data3 = 3 * np.cos(np.arange(0, num_points))
    data4 = 4 * np.cos(np.arange(0, num_points))

    # Create varying amplitudes-创建不同振幅，以此来创建一个随机波形
    amp1 = np.ones(num_points)
    amp2 = 4 + np.zeros(num_points) 
    amp3 = 2 * np.ones(num_points) 
    amp4 = 0.5 + np.zeros(num_points) 

    # 将数据合并成输入数组，其中数据对应输入，而振幅对应响应标签
    data = np.array([data1, data2, data3, data4]).reshape(num_points * 4, 1)
    amplitude = np.array([[amp1, amp2, amp3, amp4]]).reshape(num_points * 4, 1)

    return data, amplitude 

# Draw the output using the network
# 定义一个函数，用于画出将数据传入训练的神经网络后的输出
def draw_output(net, num_points_test):
    data_test, amplitude_test = create_waveform(num_points_test)
    output_test = net.sim(data_test)
    plt.plot(amplitude_test.reshape(num_points_test * 4))
    plt.plot(output_test.reshape(num_points_test * 4))

if __name__=='__main__':
    # Get data
    num_points = 30
    data, amplitude = create_waveform(num_points)

    # Create network with 2 layers
    # 创建一个两层的递归神经网络
    net = nl.net.newelm([[-2, 2]], [10, 1], [nl.trans.TanSig(), nl.trans.PureLin()])

    # Set initialized functions and init
    # 设定每层的初始化函数
    net.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
    net.layers[1].initf= nl.init.InitRand([-0.1, 0.1], 'wb')
    net.init()

    # Training the recurrent neural network
    # 训练递归神经网络
    error = net.train(data, amplitude, epochs=1000, show=100, goal=0.01)

    # Compute output from network
    # 为训练数据计算来自网络的输出
    output = net.sim(data)

    # Plot training results - 画出训练结果
    plt.subplot(211)
    plt.plot(error)
    plt.xlabel('Number of epochs')
    plt.ylabel('Error (MSE)')

    plt.subplot(212)
    plt.plot(amplitude.reshape(num_points * 4))
    plt.plot(output.reshape(num_points * 4))
    plt.legend(['Ground truth', 'Predicted output'])

    # Testing on unknown data at multiple scales
    plt.figure()

    # 创建一个随机长度的波形，查看该神经网络能否预测。在多个尺度上对未知数据进行测试
    plt.subplot(211)
    draw_output(net, 74)
    plt.xlim([0, 300])

    # 创建另一个长度更短的波形，查看该神经网络能否预测
    plt.subplot(212)
    draw_output(net, 54)
    plt.xlim([0, 300])

    plt.show()

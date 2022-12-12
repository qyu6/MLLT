import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.decomposition import PCA, FastICA

# Load data
input_file = 'mixture_of_signals.txt'
X = np.loadtxt(input_file)

# Compute ICA,创建ICA对象
ica = FastICA(n_components=4)

# Reconstruct the signals，基于ICA重构信号
signals_ica = ica.fit_transform(X)

# Get estimated mixing matrix，提取混合矩阵
mixing_mat = ica.mixing_  

# Perform PCA，执行PCA
pca = PCA(n_components=4)
signals_pca = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

# Specify parameters for output plots，定义画图参数
models = [X, signals_ica, signals_pca]

# 指定颜色
colors = ['blue', 'red', 'black', 'green']

# Plotting input signal，画出输入信号
plt.figure()
plt.title('Input signal (mixture)')
for i, (sig, color) in enumerate(zip(X.T, colors), 1):
    plt.plot(sig, color=color)

# Plotting ICA signals，画出利用ICA分离的信号
plt.figure()
plt.title('ICA separated signals')
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.94, 
        top=0.94, wspace=0.25, hspace=0.45)
# 用不同颜色画出子图
for i, (sig, color) in enumerate(zip(signals_ica.T, colors), 1):
    plt.subplot(4, 1, i)
    plt.title('Signal ' + str(i))
    plt.plot(sig, color=color)

# Plotting PCA signals  
# 画出PCA信号
plt.figure()
plt.title('PCA separated signals')
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.94, 
        top=0.94, wspace=0.25, hspace=0.45)
# 用不同的颜色画出格子图
for i, (sig, color) in enumerate(zip(signals_pca.T, colors), 1):
    plt.subplot(4, 1, i)
    plt.title('Signal ' + str(i))
    plt.plot(sig, color=color)

plt.show()


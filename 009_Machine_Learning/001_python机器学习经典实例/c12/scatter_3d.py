import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the number of values-定义生成值的个数
n = 250

# Create a lambda function to generate the random values in the given range
# 生成一个lambda函数来生成给定范围的值
f = lambda minval, maxval, n: minval + (maxval - minval) * np.random.rand(n)

# Generate the values-用函数f生成xyz的值
x_vals = f(15, 41, n)
y_vals = f(-10, 70, n)
z_vals = f(-52, -37, n)

# Plot the values
ax.scatter(x_vals, y_vals, z_vals, c='k', marker='o')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()

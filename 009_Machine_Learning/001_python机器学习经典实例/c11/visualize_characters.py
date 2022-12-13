import os
import sys

import cv2
import numpy as np

# Load input data 
input_file = 'letter.data' 

# Define visualization parameters - 定义可视化参数
scaling_factor = 10
start_index = 6
end_index = -1
h, w = 16, 8

# Loop until you encounter the Esc key - 迭代循环文件直至用户按下ESC
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = np.array([255*float(x) for x in line.split('\t')[start_index:end_index]])
        
        # 将数组重新调整为所需的形状，调整大小并将其展示
        img = np.reshape(data, (h,w))
        img_scaled = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor)
        cv2.imshow('Image', img_scaled)
        c = cv2.waitKey()
        if c == 27:
            break

import sys
import cv2
import numpy as np

# Load and display an image -- 'forest.jpg'
# input_file = sys.argv[1]
img = cv2.imread('forest.jpg')

# 显示输入图像
cv2.imshow('Original', img)

# Cropping an image
# 剪裁图像，提取输入图像的高度和宽度，然后指定边界
h, w = img.shape[:2]
start_row, end_row = int(0.21*h), int(0.73*h)
start_col, end_col= int(0.37*w), int(0.92*w)

# 用NumPy式的切分方式剪裁图像，并将其展示出来
img_cropped = img[start_row:end_row, start_col:end_col]
cv2.imshow('Cropped', img_cropped)

# Resizing an image
# 将图像大小调整为原来的1.3倍，并展示
scaling_factor = 1.3
img_scaled = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, 
        interpolation=cv2.INTER_LINEAR)
cv2.imshow('Uniform resizing', img_scaled)

# 仅在某一个维度上进行调整
img_scaled = cv2.resize(img, (250, 400), interpolation=cv2.INTER_AREA)
cv2.imshow('Skewed resizing', img_scaled)

# Save an image
# output_file = input_file[:-4] + '_cropped.jpg'
# 保存图像到输出文件
output_file = 'forest_cropped.jpg'
cv2.imwrite(output_file, img_cropped)

cv2.waitKey()

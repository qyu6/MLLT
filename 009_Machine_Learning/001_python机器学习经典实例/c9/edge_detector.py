import sys
import cv2
import numpy as np

# Load the input image -- 'chair.jpg'
# Convert it to grayscale 
# input_file = sys.argv[1]
# 加载图像，转换成灰度图
img = cv2.imread('chair.jpg', cv2.IMREAD_GRAYSCALE)

# 提取图像高度和宽度
h, w = img.shape

# 索贝尔滤波器
sobel_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)

# 运行索贝尔垂直检测器
sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

# 拉普拉斯边检测器
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# Canny边检测器
canny = cv2.Canny(img, 50, 240)

cv2.imshow('Original', img)
cv2.imshow('Sobel horizontal', sobel_horizontal)
cv2.imshow('Sobel vertical', sobel_vertical)
cv2.imshow('Laplacian', laplacian)
cv2.imshow('Canny', canny)

cv2.waitKey()


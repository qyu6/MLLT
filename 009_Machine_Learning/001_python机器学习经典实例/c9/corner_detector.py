import sys
import cv2
import numpy as np

# Load input image -- 'box.png'
# input_file = sys.argv[1]
img = cv2.imread('box.png')
cv2.imshow('Input image', img)

# 将图像转化为灰度，并将其强制转换为浮点值。浮点值将用于棱角检测过程
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = np.float32(img_gray)

# Harris corner detector 
# 对灰度图像运行哈里斯角检测器
img_harris = cv2.cornerHarris(img_gray, 7, 5, 0.04)

# Resultant image is dilated to mark the corners
# 放大图像以标记棱角
img_harris = cv2.dilate(img_harris, None)

# Threshold the image 
# 用阈值显示棱角
img[img_harris > 0.01 * img_harris.max()] = [0, 0, 0]

# 显示输出图像
cv2.imshow('Harris Corners', img)
cv2.waitKey()

import sys
import cv2
import numpy as np

# Load input image -- 'table.jpg'
# input_file = sys.argv[1]
img = cv2.imread('table.jpg')

# 将图像转为灰度
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 初始化SIFT检测器对象并提取关键点。(此处关键点并不是特征)
# sift = cv2.xfeatures2d.SIFT_create()
sift = cv2.SIFT_create()
keypoints = sift.detect(img_gray, None)

# 画出关键点
img_sift = np.copy(img)
cv2.drawKeypoints(img, keypoints, img_sift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 显示输出图像
cv2.imshow('Input image', img)
cv2.imshow('SIFT features', img_sift)
cv2.waitKey()

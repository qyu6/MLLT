import sys
import cv2
import numpy as np

# Load input image -- 'sunrise.jpg'
# input_file = sys.argv[1]
img = cv2.imread('sunrise.jpg')

# Convert it to grayscale, 转化为灰度图
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Input grayscale image', img_gray)

# Equalize the histogram
# 均衡灰度图像的直方图，并显示
img_gray_histeq = cv2.equalizeHist(img_gray)
cv2.imshow('Histogram equalized - grayscale', img_gray_histeq)

# Histogram equalization of color images
# 均衡彩色图像的直方图
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

img_histeq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

cv2.imshow('Input color image', img)
cv2.imshow('Histogram equalized - color', img_histeq)

cv2.waitKey()




import cv2
import numpy as np

# Load face, eye, and nose cascade files
# 加载人脸,眼睛和鼻子级联文件
face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_mcs_nose.xml')

# Check if face cascade file has been loaded
# 检查脸部级联文件是否加载
if face_cascade.empty():
	raise IOError('Unable to load the face cascade classifier xml file')

# Check if eye cascade file has been loaded
# 检查眼睛级联文件是否加载
if eye_cascade.empty():
	raise IOError('Unable to load the eye cascade classifier xml file')

# Check if nose cascade file has been loaded
# 检查鼻子级联文件是否加载
if nose_cascade.empty():
    raise IOError('Unable to load the nose cascade classifier xml file')

# Initialize video capture object and define scaling factor
# 初始化视频采集对象并定义比例系数
cap = cv2.VideoCapture(0)

# 定义比例系数
scaling_factor = 1

print('start detecting...')

while True:
    # Read current frame, resize it, and convert it to grayscale
    ret, frame = cap.read()

    # 调整帧的大小
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, 
            interpolation=cv2.INTER_AREA)

    # 将图像转为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run face detector on the grayscale image
    # 在灰度图上运行人脸检测器
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Run eye and nose detectors within each face rectangle
    # 在每张脸的矩形区域运行眼睛和鼻子检测器
    for (x,y,w,h) in faces:
        
        # Grab the current ROI in both color and grayscale images
        # 从彩色与灰度图中提取人脸的ROI信息
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Run eye detector in the grayscale ROI
        # 在灰度图ROI信息中检测眼睛
        eye_rects = eye_cascade.detectMultiScale(roi_gray)

        # Run nose detector in the grayscale ROI
        # 在灰度图ROI信息中检测鼻子
        nose_rects = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)

        # Draw green circles around the eyes
        # 在眼睛周围画绿色的圈
        for (x_eye, y_eye, w_eye, h_eye) in eye_rects:
            center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
            radius = int(0.3 * (w_eye + h_eye))
            color = (0, 255, 0)
            thickness = 3
            cv2.circle(roi_color, center, radius, color, thickness)

        # 在鼻子周围画矩形
        for (x_nose, y_nose, w_nose, h_nose) in nose_rects:
            cv2.rectangle(roi_color, (x_nose, y_nose), (x_nose+w_nose, 
                y_nose+h_nose), (0,255,0), 3)
            break
    
    # Display the image，展示图像
    cv2.imshow('Eye and nose detector', frame)

    # Check if Esc key has been pressed，在下次迭代之前等待1ms，按下ESC跳出循环
    c = cv2.waitKey(1)
    if c == 27:
        break

# Release video capture object and close all windows
# 释放视频采样对象并关闭窗口
cap.release()
cv2.destroyAllWindows()


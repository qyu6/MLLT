import cv2
import numpy as np

# Load the face cascade file
# 导入人脸检测级联文件,可用作检测器的训练模型
face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')

# Check if the face cascade file has been loaded
# 确定级联文件是否正确地加载
if face_cascade.empty():
	raise IOError('Unable to load the face cascade classifier xml file')

# Initialize the video capture object
# 初始化视频采集对象
cap = cv2.VideoCapture(0)
print('preparing...')

# Define the scaling factor-定义图像向下采样的比例系数. 0.5
scaling_factor = 1.5

# Loop until you hit the Esc key-循环采集直到按下ESC
while True:

    # Capture the current frame and resize it-采样当前帧并进行调整
    ret, frame = cap.read()

    # 调整帧的大小
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, 
            interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    # 将图像转为灰度图. 来运行人脸检测器
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run the face detector on the grayscale image
    # 在灰度图像上运行人脸检测器.
    # 1.3 - 指每个阶段的乘积系数
    # 5 - 指每个候选矩阵应该拥有的最小近邻数量,这样可以维持这一数量
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles on the image
    # 检测到的人脸区域,脸部画出矩形
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

    # Display the image-展示输出图像
    cv2.imshow('Face Detector', frame)

    # Check if Esc key has been pressed
    c = cv2.waitKey(1)
    if c == 27:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

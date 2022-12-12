import os

import cv2
import numpy as np
from sklearn import preprocessing

# Class to handle tasks related to label encoding
# 定义一个类来处理与类标签编码相关的所有任务
class LabelEncoder(object):

    # Method to encode labels from words to numbers
    # (定义一个方法来为这些标签编码。输入训练数据，标签用单词表示，但需要数据来训练系统)
    # 该方法将定义一个预处理对象，该对象将单词转换成数字，同时保留这种前向后向的映射关系
    # 将单词转化成数字的编码方法。
    def encode_labels(self, label_words):
        self.le = preprocessing.LabelEncoder()
        self.le.fit(label_words)

    # Convert input label from word to number
    # 将输入单词转化成数字
    def word_to_num(self, label_word):
        return int(self.le.transform([label_word])[0])

    # Convert input label from number to word
    # 将数字转换成原始单词
    def num_to_word(self, label_num):
        return self.le.inverse_transform([label_num])[0]

# Extract images and labels from input path
# 定义一个方法，用于从输入文件夹中提取图像和标签
def get_images_and_labels(input_path):
    label_words = []

    # Iterate through the input path and append files
    # 对输入文件夹做递归迭代，提取所有图像的路径
    for root, dirs, files in os.walk(input_path):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename)
            label_words.append(filepath.split('/')[-2]) 
            
    # Initialize variables-初始化变量
    images = []
    le = LabelEncoder()
    le.encode_labels(label_words)
    labels = []

    # Parse the input directory-为训练解析输入目录
    for root, dirs, files in os.walk(input_path):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename)

            # Read the image in grayscale format
            # 将当前图像读取成灰度格式
            image = cv2.imread(filepath, 0) 

            # Extract the label
            # 从文件夹路径中提取标签
            name = filepath.split('/')[-2]
                
            # Perform face detection
            # 对该图像做人脸检测
            faces = faceCascade.detectMultiScale(image, 1.1, 2, minSize=(100,100))

            # Iterate through face rectangles
            # 循环处理每一张脸，提取ROI属性值
            for (x, y, w, h) in faces:
                images.append(image[y:y+h, x:x+w])
                labels.append(le.word_to_num(name))

    return images, labels, le

if __name__=='__main__':
    cascade_path = "cascade_files/haarcascade_frontalface_alt.xml"
    path_train = 'faces_dataset/train'
    path_test = 'faces_dataset/test'

    # Load face cascade file-加载人脸级联文件
    faceCascade = cv2.CascadeClassifier(cascade_path)

    # Initialize Local Binary Patterns Histogram face recognizer
    # 生成局部二值模式直方图人脸识别器对象
    # recognizer = cv2.face.createLBPHFaceRecognizer()
    recognizer = cv2.face.LBPHFaceRecognizer()

    # Extract images, labels, and label encoder from training dataset
    # 为输入路径提取图像、标签和标签编码器
    images, labels, le = get_images_and_labels(path_train)

    # Train the face recognizer 
    # 用提取的数据训练人脸识别器
    print("\nTraining...")
    recognizer.train(images, np.array(labels))

    # Test the recognizer on unknown images
    # 用未知数据测试人脸识别器
    print ('\nPerforming prediction on test images...')
    stop_flag = False
    for root, dirs, files in os.walk(path_test):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename)

            # Read the image,读取图像
            predict_image = cv2.imread(filepath, 0)

            # Detect faces，检测人脸
            faces = faceCascade.detectMultiScale(predict_image, 1.1, 
                    2, minSize=(100,100))

            # Iterate through face rectangles
            # 循环处理每张脸，对每个人脸ROI,运行人脸识别器
            for (x, y, w, h) in faces:
                # Predict the output
                predicted_index, conf = recognizer.predict(
                        predict_image[y:y+h, x:x+w])

                # Convert to word label
                # 将标签转化为单词
                predicted_person = le.num_to_word(predicted_index)

                # Overlay text on the output image and display it
                # 在输出图像中叠加文字，并显示图像
                cv2.putText(predict_image, 'Prediction: ' + predicted_person, 
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 6)
                cv2.imshow("Recognizing face", predict_image)

            # 如果没有按下ESC，持续循环
            c = cv2.waitKey(0)
            if c == 27:
                stop_flag = True
                break

        if stop_flag:
            break


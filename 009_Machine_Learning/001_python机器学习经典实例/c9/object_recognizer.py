import argparse 
# import cPickle as pickle 
import pickle

import cv2
import numpy as np

import build_features as bf
from trainer import ERFTrainer

# 定义一个参数解析器
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Extracts features \
            from each line and classifies the data')
    parser.add_argument("--input-image", dest="input_image", required=True,
            help="Input image to be classified")
    parser.add_argument("--model-file", dest="model_file", required=True,
            help="Input file containing the trained model")
    parser.add_argument("--codebook-file", dest="codebook_file", 
            required=True, help="Input file containing the codebook")
    return parser

# 定义一个类来处理图像标签提取函数
class ImageTagExtractor(object):
    def __init__(self, model_file, codebook_file):
        with open(model_file, 'r') as f:
            self.erf = pickle.load(f)

        with open(codebook_file, 'r') as f:
            self.kmeans, self.centroids = pickle.load(f)

    # 定义一个函数,用于使用训练好ERF模型来预测输出
    def predict(self, img, scaling_size):
        img = bf.resize_image(img, scaling_size)
        feature_vector = bf.BagOfWords().construct_feature(
                img, self.kmeans, self.centroids)
        image_tag = self.erf.classify(feature_vector)[0]
        return image_tag

# 定义main函数,加载输入图像
if __name__=='__main__':
    args = build_arg_parser().parse_args()
    model_file = args.model_file
    codebook_file = args.codebook_file
    input_image = cv2.imread(args.input_image)
        
    # 合理调整图像大小 
    scaling_size = 200
    
    # 命令行打印输出结果
    print ("\nOutput:", ImageTagExtractor(model_file, 
            codebook_file).predict(input_image, scaling_size))

# 命令行输入如下命令来启动:
# $python object_recognizer.py --input-image imagefile.jpb --model-file erf.pkl --codebook-file codebook.pkl
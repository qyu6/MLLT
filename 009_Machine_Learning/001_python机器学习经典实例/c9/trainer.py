import argparse 
import cPickle as pickle 

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing

# 定义一个参数解析器
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Trains the classifier')
    parser.add_argument("--feature-map-file", dest="feature_map_file", required=True,
            help="Input pickle file containing the feature map")
    parser.add_argument("--model-file", dest="model_file", required=False,
            help="Output file where the trained model will be stored")
    return parser

# 定义一个类来处理ERF训练.用到一个标签编码器来训练标签进行编码
class ERFTrainer(object):
    def __init__(self, X, label_words):
        self.le = preprocessing.LabelEncoder()  
        self.clf = ExtraTreesClassifier(n_estimators=100, 
                max_depth=16, random_state=0)

        # 对标签编码并训练分类器
        y = self.encode_labels(label_words)
        self.clf.fit(np.asarray(X), y)

    # 定义一个函数,对标签进行编码
    def encode_labels(self, label_words):
        self.le.fit(label_words) 
        return np.array(self.le.transform(label_words), dtype=np.float32)

    # 定义一个函数,用于将未知数据点进行分类
    def classify(self, X):
        label_nums = self.clf.predict(np.asarray(X))
        label_words = self.le.inverse_transform([int(x) for x in label_nums]) 
        return label_words

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    feature_map_file = args.feature_map_file
    model_file = args.model_file

    # Load the feature map.加载build_features.py中生成的特征地图
    with open(feature_map_file, 'r') as f:
        feature_map = pickle.load(f)

    # Extract feature vectors and the labels-提取特征向量和标记
    label_words = [x['object_class'] for x in feature_map]
    dim_size = feature_map[0]['feature_vector'].shape[1]  
    X = [np.reshape(x['feature_vector'], (dim_size,)) for x in feature_map]
    
    # Train the Extremely Random Forests classifier-基于训练数据训练ERF
    erf = ERFTrainer(X, label_words) 

    # 保存训练的ERF模型
    if args.model_file:
        with open(args.model_file, 'w') as f:
            pickle.dump(erf, f)


# 在命令行运行代码来启动:
# $python trainer.py --feature-map-file feature_map.pkl --model-file erf.pkl

# 结果生成了一个erf.pkl文件
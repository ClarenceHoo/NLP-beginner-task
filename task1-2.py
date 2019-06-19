'''
实验：
用ngram来分类
其中ngram为单一数据训练模型
ngram-batch为带batch的ngram训练模型
训练取了10epoch
训练好后将原始数据通过ngram得到的特征向量保存到了./data/ngram_featrue_x.npy
'''
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd

x_train = np.array(np.load('./data/ngram_featrue_x.npy'))
y_train = np.load('./data/y_train.npy')
print(x_train.shape)
print(y_train.shape)

logist = LogisticRegression()
logist.fit(x_train,y_train)

predicted = logist.predict(x_train)
print(np.mean(predicted == y_train))

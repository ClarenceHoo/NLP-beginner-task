
''' 
实现基于logistic/softmax regression的文本分类

参考

文本分类
《神经网络与深度学习》 第2/3章
数据集：Classify the sentiment of sentences from the Rotten Tomatoes dataset

实现要求：NumPy

需要了解的知识点：

文本特征表示：Bag-of-Word，N-gram
分类器：logistic/softmax regression，损失函数、（随机）梯度下降、特征选择
数据集：训练集/验证集/测试集的划分
实验：
用CountVectorizer来分类
分析不同的特征、损失函数、学习率对最终分类性能的影响
shuffle 、batch、mini-batch
时间：两周
'''
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import  CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pandas as pd

	
	
def task_without_CountVectorizer():
####(156060,)自写CountVectorizer
####最终结果0.5219466871716006
	df = pd.read_csv('train.tsv',header=0,delimiter='\t')
	x_train = df['Phrase']
	y_train = df['Sentiment']
	all=[]

	for i in range(len(x_train)):
		all.extend(x_train[i])
	voc = set(all)
	x_train_idx = []
	for i in range(len(x_train)):
		tmp = np.zeros(len(voc))
		for j, word in enumerate(voc):
			tmp[j] = x_train[i].count(word)	
		x_train_idx.append(tmp)
	x_train_id = np.array(x_train_idx)
	#np.save('./data/x_train.npy',x_train_id)
	#np.save('./data/y_train.npy',y_train)
	#x_train = np.load('./data/x_train.npy')
	#y_train = np.load('./data/y_train.npy')
	logist = LogisticRegression()
	logist.fit(x_train,y_train)
	x_test = x_train
	predicted = logist.predict(x_test)
	print(np.mean(predicted == y_train))
def task_with_CountVectorizer():
####自带的CountVectorizer
####最终结果为0.6962770729206715
	df = pd.read_csv('train.tsv',header=0,delimiter='\t')
	x_train = df['Phrase']
	y_train = df['Sentiment']
	count_vec = CountVectorizer()
	x_count_train = count_vec.fit_transform(x_train)
	logist = LogisticRegression()
	logist.fit(x_count_train,y_train)
	x_test = x_count_train
	predicted = logist.predict(x_test)
	print(np.mean(predicted == y_train))

task_with_CountVectorizer()

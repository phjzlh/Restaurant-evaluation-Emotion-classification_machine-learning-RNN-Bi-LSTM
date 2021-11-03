import os
import re
import jieba
import nltk
import time
import warnings
import numpy as np
from collections import defaultdict
import math
import operator
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import label_binarize

from sklearn.ensemble import  AdaBoostClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

t0 = time.time()
def get_txt_data(txt_file):
    mostwords=[]
    file=open(txt_file, 'r', encoding='utf-8')
    for line in file.readlines():
        curline=line.strip().split()
        mostwords.append(curline)
    return mostwords


train_doc1 = get_txt_data('train_x.txt')
label_doc1 = get_txt_data('train_y.txt')
train_doc2 = get_txt_data('test_x.txt')
label_doc2 = get_txt_data('test_y.txt')

train_doc = train_doc1 + train_doc2
label_doc = label_doc1 + label_doc2

t1 = time.time()
print("文件载入时间：", t1-t0)
#将字符label转换为数字label
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
label = le.fit_transform(label_doc)
y = label

#删除停用词
def context_cut(sentence):
    words_str = []
    words_list = []
    #获取停用词
    stop=open('ChineseStopWords.txt','r+',encoding='utf-8')
    stopwords=stop.read().split('\n')
    for word in sentence:
        if not(word in stopwords):
            words_list.append(word)
        words_str=','.join(words_list)
    return words_str , words_list
words=[]
for i in range(len(train_doc)):
    cut_words_str, cut_words_list = context_cut(train_doc[i])
    words.append(cut_words_list)

# 将分词重新组合成句子
senwords=[]
for i in range(len(words)):
    senwords.append(" ".join(words[i]))
t2 = time.time()
print("删除停用词时间：", t2-t1)
# 进行tf-idf权重计算
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
x_train = senwords
vectorizer = CountVectorizer(max_features=10000)
tf_idf_transformer = TfidfTransformer()
tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(x_train))
x_train_weight = tf_idf.toarray()
X_train, X_test, y_train, y_test = train_test_split(x_train_weight, y, test_size=0.15, random_state=0)
print('训练集,label尺寸：{},{}, 测试集,label尺寸：{},{}'.format(len(X_train), len(y_train), len(X_test), len(y_test)))
pos = 0
neg = 0
for i in y_test:
    if i == 1:
        pos +=1
    else:
        neg +=1
print("训练集pos个数：{}   neg个数：{}".format(pos, neg))

t3 = time.time()
print("生成特征向量时间：", t3-t2)

print("预处理结束，开始训练。")

Classifier_str = ['LogisticRegression()', 'MultinomialNB()',
                  'KNeighborsClassifier()', 'RandomForestClassifier()',
                  'DecisionTreeClassifier()', 'AdaBoostClassifier()', 'SVC()']

for c in Classifier_str:
    #print("开始进行{}训练".format(c))
    t0 = time.time()
    # 这个是返回的index集合，i是所有训练集的index，j是所有测试集的index
    clf = eval(c)
    clf.fit(X_train, y_train)
    scores = clf.score(X_test, y_test)
    y_score = clf.predict(X_test)
    pos_pre = 0
    neg_pre = 0
    tp = 0
    tn =0
    fp = 0
    fn = 0
    for i in y_score:
        if i == 1:
            pos_pre += 1
        if i == 0:
            neg_pre += 1
    for i in range(len(y_test)):
        if y_test[i] == 1:
            if y_score[i] == 1:
                tp +=1
            if y_score[i] == 0:
                fn +=1
        if y_test[i] ==0:
            if y_score[i] == 1:
                fp +=1
            if y_score[i] == 0:
                tn +=1

    pre = precision_score(y_test, y_score, average=None)
    rec = recall_score(y_test, y_score, average=None)
    f1 = f1_score(y_test, y_score, average=None)
    t1 = time.time()
    print("{}训练消耗时间：{}".format(c, t1 - t0))
    print("预测为正例的个数:{}   负例的个数：{}".format(pos_pre, neg_pre))
    print("tp： {} fp： {}\nfn： {} tn ： {}".format(tp, fp, fn, tn))
    print("{}准确率： {}\n精确率： {}\n召回率： {}\nF1_scores ： {}\n".format(c, scores, pre, rec, f1))



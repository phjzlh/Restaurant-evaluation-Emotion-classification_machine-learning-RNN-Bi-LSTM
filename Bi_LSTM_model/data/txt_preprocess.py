# 创建时间：2021/10/28 9:45
import pandas as pd
import string
import jieba
import os
import numpy as np
from sklearn.model_selection import train_test_split

csvfile_path = os.path.join(r'D:/BaiduNetdiskDownload/yf_dianping/ratings', 'ratings.csv')


def getdata(file_path):
    data = pd.read_csv(file_path, nrows=1000000)
    print('csv文件读取完成')
    data = data.dropna()
    data.drop(['userId'], axis=1, inplace=True)
    data.drop(['restId'], axis=1, inplace=True)
    data.drop(['rating_env'], axis=1, inplace=True)
    data.drop(['rating_flavor'], axis=1, inplace=True)
    data.drop(['rating_service'], axis=1, inplace=True)
    data.drop(['timestamp'], axis=1, inplace=True)

    labels_index = data.loc[:, 'rating']
    comments_index = data.loc[:, 'comment']

    labels = labels_index.values
    comments = comments_index.values
    return labels, comments


def data_split(comments, labels):
    train_x, test_x, train_y, test_y = train_test_split(labels, comments, test_size=0.2, random_state=0)
    return train_x, test_x, train_y, test_y


def array2txt(tr_comments, test_comments, tr_labels, test_labels):
    punc = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·"""
    dictp = {i: '' for i in punc}
    punc_table = str.maketrans(dictp)
    with open('ChineseStopWords.txt', 'r', encoding='utf-8') as sf:
        stopwords = [line.strip() for line in sf.readlines()]

    with open('train_x.txt', 'w', encoding='utf-8') as f:
        for i in range(len(tr_comments) - 1):
            word_list = []
            temp = jieba.cut(str(tr_comments[i]).replace('\t', '').replace('\n', '').replace('\r', '').replace('\xa0', '').replace(' ', '').translate(punc_table))
            for word in temp:
                if word not in stopwords:
                    word_list.append(word)
            sentense = ' '.join(word_list)
            f.write(sentense + '\n')
        word_list = []
        temp = jieba.cut(str(tr_comments[i + 1]).replace('\t', '').replace('\n', '').replace('\r', '').replace('\xa0', '').replace(' ', '').translate(punc_table))
        for word in temp:
            if word not in stopwords:
                word_list.append(word)
        senten = ' '.join(word_list)
        f.write(senten)

    with open('test_x.txt', 'w', encoding='utf-8') as f:
        for i in range(len(test_comments) - 1):
            word_list = []
            temp = jieba.cut(str(test_comments[i]).replace('\t', '').replace('\n', '').replace('\r', '').replace('\xa0', '').replace(' ', '').translate(punc_table))
            for word in temp:
                if word not in stopwords:
                    word_list.append(word)
            sentense = ' '.join(word_list)
            f.write(sentense + '\n')
        word_list = []
        temp = jieba.cut(str(test_comments[i + 1]).replace('\t', '').replace('\n', '').replace('\r', '').replace('\xa0', '').replace(' ', '').translate(punc_table))
        for word in temp:
            if word not in stopwords:
                word_list.append(word)
        senten = ' '.join(word_list)
        f.write(senten)

    count = {'tr_pos': 0, 'tr_neg': 0, 'test_pos': 0, 'test_neg': 0}

    with open('train_y.txt', 'w', encoding='utf-8') as f:
        for i in range(len(tr_labels) - 1):
            if int(tr_labels[i]) >= 3:
                f.write('1' + '\n')
                count['tr_pos'] += 1
            else:
                f.write('0' + '\n')
                count['tr_neg'] += 1
        if int(tr_labels[i + 1]) >= 3:
            f.write('1')
            count['tr_pos'] += 1
        else:
            f.write('0')
            count['tr_neg'] += 1

    with open('test_y.txt', 'w', encoding='utf-8') as f:
        for i in range(len(test_labels) - 1):
            if int(test_labels[i]) >= 3:
                f.write('1' + '\n')
                count['test_pos'] += 1
            else:
                f.write('0' + '\n')
                count['test_neg'] += 1
        if int(test_labels[i + 1]) >= 3:
            f.write('1')
            count['test_pos'] += 1
        else:
            f.write('0')
            count['test_neg'] += 1

    print(count)


if __name__ == '__main__':
    comments, labels = getdata(csvfile_path)
    train_x, test_x, train_y, test_y = data_split(comments, labels)
    array2txt(train_x, test_x, train_y, test_y)

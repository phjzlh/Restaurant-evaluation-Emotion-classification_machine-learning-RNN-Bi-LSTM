# -*-coding:utf-8-*-
# 创建时间：2021/10/30 14:46
import jieba
import torch
from LSTMmodel import LSTMModel
import pickle
import dataset
from torch.utils.data import DataLoader
import vocab
from tqdm import tqdm

voc_model = pickle.load(open('./models/vocab.pkl', 'rb'))


def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_vocab(sentence):
    seg_list = jieba.lcut(sentence, cut_all=False)
    vocab_model = pickle.load(open('./models/vocab.pkl', 'rb'))
    result = vocab_model.transform(seg_list, 100)
    return result


if __name__ == '__main__':
    print('输入评论：')
    sentence = str(input())
    vocab = get_vocab(sentence)
    input = torch.tensor(vocab).to(device())
    input = input.reshape(1, -1)
    lstm_model_pred = LSTMModel().to(device())
    lstm_model_pred.load_state_dict(torch.load('./modelDict/model.pth'), strict=True)

    output = lstm_model_pred(input)
    pred = output.argmax(1).item()
    if pred == 1:
        print('该评论为好评')
    else:
        print('该评论为差评')
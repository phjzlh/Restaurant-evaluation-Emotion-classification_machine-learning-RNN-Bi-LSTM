# -*-coding:utf-8-*-
# 创建时间：2021/10/29 10:26
import os
from vocab import vocab
import pickle
import re
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class SentimentDataset(Dataset):
    def __init__(self, train=True):
        self.comment_data_path = r'./data'
        self.comment_data_path += r'/train_x_new.txt' if train else r'/test_x_new.txt'
        self.label_data_path = r'./data'
        self.label_data_path += r'/train_y_new.txt' if train else r'/test_y_new.txt'
        self.comment, self.label = self.get_comment_label()

    def __getitem__(self, idx):
        return self.comment[idx], self.label[idx]

    def __len__(self):
        return len(self.comment)

    def get_comment_label(self):
        comment = []
        label = []
        with open(self.label_data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                label.append(int(line[0]))
        with open(self.comment_data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line_words = re.split(r'[\s]', line)
                comment.append(line_words)
        return comment, label


def mycollate_fn(batch):
    comments, labels = zip(*batch)
    return comments, labels


if __name__ == '__main__':
    dzdp_dataset = SentimentDataset(True)
    my_dataloader = DataLoader(dzdp_dataset, batch_size=1, shuffle=True, collate_fn=mycollate_fn)
    for comments, label in my_dataloader:
        vocab_model = pickle.load(open('./models/vocab.pkl', 'rb'))
        print(label)
        result = vocab_model.transform(comments[0], 100)
        print(result)
        break
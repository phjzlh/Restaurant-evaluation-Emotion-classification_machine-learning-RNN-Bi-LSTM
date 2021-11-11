# -*-coding:utf-8-*-
# 创建时间：2021/10/29 11:04
import os
import pickle
from vocab import vocab
import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def collate_fn(batch):
    reviews, labels = zip(*batch)
    return reviews, labels


def get_dataloader(train=True):
    dzdp_dataset = dataset.SentimentDataset(train)
    my_dataloader = DataLoader(dzdp_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    return my_dataloader


if __name__ == '__main__':
    ws = vocab()
    dl_train = get_dataloader(True)
    dl_test = get_dataloader(False)
    for comments, labels in tqdm(dl_train, total=len(dl_train)):
        for sentence in comments:
            ws.fit(sentence)
    # for comments, labels in tqdm(dl_test, total=len(dl_test)):
    #     for sentence in comments:
    #         ws.fit(sentence)
    ws.build_vocab()
    print(len(ws))
    pickle.dump(ws, open('./models/vocab.pkl', 'wb'))

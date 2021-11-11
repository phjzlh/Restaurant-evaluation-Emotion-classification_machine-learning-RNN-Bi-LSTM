# -*-coding:utf-8-*-
# 创建时间：2021/10/29 15:16
import torch


class ConfuseMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.confuse_mat = torch.zeros(2, 2)
        self.tn = self.confuse_mat[0, 0]
        self.fn = self.confuse_mat[0, 1]
        self.tp = self.confuse_mat[1, 1]
        self.fp = self.confuse_mat[1, 0]
        self.acc = 0
        self.pre = 0
        self.rec = 0
        self.F1 = 0

    def update(self, pred, label):
        for l, p in zip(label.view(-1), pred.view(-1)):
            self.confuse_mat[l.long(), p.long()] += 1
        self.tn = self.confuse_mat[0, 0]
        self.fn = self.confuse_mat[0, 1]
        self.tp = self.confuse_mat[1, 1]
        self.fp = self.confuse_mat[1, 0]

        self.acc = (self.tn + self.tp) / self.confuse_mat.sum()
        self.pre = self.tp / (self.tp + self.fp)
        self.rec = self.tp / (self.tp + self.fn)
        self.F1 = 2 * self.pre * self.rec / (self.pre + self.rec)
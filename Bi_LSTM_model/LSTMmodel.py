# -*-coding:utf-8-*-
# 创建时间：2021/10/29 13:20
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from tqdm import tqdm
import dataset
import vocab
import shutil
from confuseMeter import ConfuseMeter
import os
from torch.utils.tensorboard import SummaryWriter

train_batchsize = 32
test_batchsize = 16
voc_model = pickle.load(open('./models/vocab.pkl', 'rb'))
sequence_max_len = 100


def collate_fn(batch):
    comments, labels = zip(*batch)
    comments = torch.LongTensor([voc_model.transform(i, max_len=sequence_max_len) for i in comments])
    labels = torch.LongTensor(labels)
    return comments, labels


def get_dataloader(train = True):
    dzdp_dataset = dataset.SentimentDataset(train)
    batchsize = train_batchsize if train else test_batchsize
    return DataLoader(dzdp_dataset, batch_size=batchsize, shuffle=True, collate_fn=collate_fn)


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(voc_model), embedding_dim=200, padding_idx=voc_model.PAD).to()
        self.embedding.weight.requires_grad = True
        self.lstm = nn.LSTM(input_size=200, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True,
                            dropout=0.5)
        self.ln1 = nn.LayerNorm(64 * 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 2, 32)
        self.ln2 = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, input):
        input_embeded = self.embedding(input)
        output, (h_n, c_n) = self.lstm(input_embeded)
        out = torch.cat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1)
        out = self.ln1(out)
        out = self.fc1(out)
        out = self.ln2(out)
        out = torch.tanh(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=-1)


def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def train(lstm_model, epoch):
    train_dataloader = get_dataloader(train=True)
    optimizer = Adam(lstm_model.parameters(), lr=0.001, )
    for i in range(epoch):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        for idx, (data, target) in enumerate(bar):
            optimizer.zero_grad()
            data = data.to(device())
            target = target.to(device())
            output = lstm_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            bar.set_description('epoch:{} idx:{} loss:{:.6f}'.format(i, idx, loss.item()))



def test(lstm_model):
    test_loss = 0
    correct = 0
    lstm_model.eval()
    test_dataloader = get_dataloader(False)
    confuse_meter = ConfuseMeter()
    with torch.no_grad():
        for data, target in tqdm(test_dataloader):
            data = data.to(device())
            target = target.to(device())
            output = lstm_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            confuse_meter.update(pred, target)
    test_loss /= len(test_dataloader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))
    print(confuse_meter.confuse_mat)
    print('\nPrecision: {:.4f}, Recall: {:.4f}, F1: {:.4f}\n'.format(
        confuse_meter.pre, confuse_meter.rec, confuse_meter.F1))
    return confuse_meter


if __name__ == '__main__':

    lstm_model = LSTMModel().to(device())
    train(lstm_model, 5)
    torch.save(lstm_model.state_dict(), './modelDict/model.pth')
    lstm_model_test = LSTMModel().to(device())
    lstm_model_test.load_state_dict(torch.load('./modelDict/model.pth'), strict=True)
    test(lstm_model_test)


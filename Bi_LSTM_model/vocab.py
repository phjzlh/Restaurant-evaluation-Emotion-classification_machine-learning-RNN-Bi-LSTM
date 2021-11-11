# -*-coding:utf-8-*-
# 创建时间：2021/10/29 11:05


class vocab:
    UNK_TAG = '<unk>'
    PAD_TAG = '<pad>'
    PAD = 0
    UNK = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {}

    def fit(self, sentence):
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min_count=20, max_count=None, max_features=None):
        if min_count is not None:
            self.count = {word: count for word, count in self.count.items() if count >= min_count}
        if max_count is not None:
            self.count = {word: count for word, count in self.count.items() if count <= max_count}
        if max_features is not None:
            self.count = dict(sorted(self.count.items(), lambda x: x[-1], reverse=True)[:max_features])

        for word in self.count:
            self.dict[word] = len(self.dict)

        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        if len(sentence) > max_len:
            sentence = sentence[:max_len]
        else:
            sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))  # 填充PAD

        return [self.dict.get(i, 1) for i in sentence]

    def inverse_transform(self, incides):
        return [self.inverse_dict.get(i, "<UNK>") for i in incides]

    def __len__(self):
        return len(self.dict)


if __name__ == '__main__':
    sentences = [["今天", "天气", "很", "好"],
                 ["今天", "去", "吃", "什么"]]
    ws = vocab()
    for sentence in sentences:
        # 统计词频
        ws.fit(sentence)
    # 构造词典
    ws.build_vocab(min_count=1)
    print(ws.dict)
    print(ws.count)
    # 把句子转换成数字序列
    ret = ws.transform(["好", "好", "好", "好", "好", "好", "好", "热", "呀"], max_len=13)
    print(ret)
    # 把数字序列转换成句子
    ret = ws.inverse_transform(ret)
    print(ret)
    pass
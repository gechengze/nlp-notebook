import torch
import torchtext
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import pandas as pd
import re
import jieba
import numpy as np
from tqdm.notebook import tqdm
from collections import Counter


# 构造数据集
class MyDataset(Dataset):
    def __init__(self, file_path, tokenizer, stopwords, debug=True):
        df = pd.read_csv(file_path)
        df = df.dropna().reset_index(drop=True)
        if debug:
            df = df.sample(2000).reset_index(drop=True)
        else:
            df = df.sample(50000).reset_index(drop=True)
        # 读取常用停用词
        counter = Counter()
        sentences = []
        for title in tqdm(df['title']):
            # 去除标点符号
            title = re.sub(r'[^\u4e00-\u9fa5]', '', title)
            tokens = [token for token in tokenizer(title.strip()) if token not in stopwords]
            counter.update(tokens)
            sentences.append(tokens)
        self.vocab = torchtext.vocab.vocab(counter, specials=['<unk>', '<pad>'])

        # 构造输入和输出，跳元模型，用当前字预测前一个字和后一个字
        self.inputs = []
        self.labels = []
        for sen in sentences:
            for i in range(1, len(sen) - 1):
                self.inputs.append([self.vocab[sen[i]]])
                self.labels.append([self.vocab[sen[i - 1]]])
                self.inputs.append([self.vocab[sen[i]]])
                self.labels.append([self.vocab[sen[i + 1]]])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.LongTensor(self.inputs[idx]), torch.LongTensor(self.labels[idx])


file_path = '../data/THUCNews/train.csv'
tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='zh_core_web_sm')
stopwords = [line.strip() for line in open('../stopwords/cn_stopwords.txt', 'r', encoding='utf-8').readlines()]
dataset = MyDataset(file_path, tokenizer, stopwords)
dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        # W和WT的形状是转置的
        self.W = nn.Embedding(vocab_size, embed_size)  # vocab_size -> embed_size
        self.WT = nn.Linear(embed_size, vocab_size, bias=False)  # embed_size -> vocab_size

    def forward(self, X):
        # X形状：batch_size * vocab_size
        hidden_layer = self.W(X)
        output_layer = self.WT(hidden_layer)
        return output_layer


# 初始化模型
model = Word2Vec(vocab_size=len(dataset.vocab), embed_size=512)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 查看模型
print(model)

# 训练20个epoch
for epoch in range(20):
    for train_input, train_label in dataloader:
        output = model(train_input)
        loss = criterion(output.squeeze_(), train_label.squeeze_())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch:', epoch + 1, 'loss =', '{:.6f}'.format(loss))

W, WT = model.parameters()
# W对应vocab中每个词的vector，这里是512维
print(W.shape, WT.shape)
print(W[0])

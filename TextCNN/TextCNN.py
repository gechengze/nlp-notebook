import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import jieba
import re


class MyDataset(Dataset):
    def __init__(self, max_len, debug=True):
        super().__init__()
        df = pd.read_csv('../../datasets/THUCNews/train.csv')
        df = df.dropna()
        if debug:
            df = df.sample(2000).reset_index(drop=True)
        else:
            df = df.sample(50000).reset_index(drop=True)
        # 读取常用停用词
        stopwords = [line.strip() for line in open('../../stopwords/cn_stopwords.txt', 'r', encoding='utf-8').readlines()]
        sentences = []
        for title in df['title']:
            # 去除标点符号
            title = re.sub(r'[^\u4e00-\u9fa5]', '', title)
            # jieba分词
            sentence_seged = jieba.cut(title.strip())
            outstr = ''
            for word in sentence_seged:
                if word != '\t' and word not in stopwords:
                    outstr += word
                    outstr += ' '
            if outstr != '':
                sentences.append(outstr)
        # 获取所有词(token), <pad>用来填充不满足max_len的句子
        token_list = ['<pad>'] + list(set(' '.join(sentences).split()))
        # token和index互转字典
        self.token2idx = {token: i for i, token in enumerate(token_list)}
        self.idx2token = {i: token for i, token in enumerate(token_list)}
        self.vocab_size = len(self.token2idx)

        self.inputs = []
        for sentence in sentences:
            tokens = sentence.split()
            input_ = [self.token2idx[token] for token in tokens]
            if len(input_) < max_len:
                self.inputs.append(input_ + [self.token2idx['<pad>']] * (max_len - len(input_)))
            else:
                self.inputs.append(input_[: max_len])

        self.labels = [[label] for label in df['label'].values.tolist()]
        self.n_class = len(df['label'].unique())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.LongTensor(self.inputs[idx]), torch.LongTensor(self.labels[idx])


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, n_class):
        super().__init__()
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 输入通道为1，卷成16通道输出，卷积核大小为(3*embed_size)，3类似于n-gram，可以换
        self.conv = nn.Conv2d(1, 16, (3, embed_size))
        self.dropout = nn.Dropout(0.2)
        # 输出头
        self.fc = nn.Linear(16, n_class)

    def forward(self, x):  # x: [batch_size * 句子长度]
        x = self.embedding(x)  # [batch_size * 句子长度 * embed_size]
        x = x.unsqueeze(1)  # [batch_size * 1 * 句子长度 * embed_size]，加一个维度，用于卷积层的输入
        x = self.conv(x)  # [batch_size * 16(卷积层输出通道数) * 8(卷积后的宽) * 1(卷积后的高)]
        x = x.squeeze(3)  # [batch_size * 16(卷积层输出通道数) * 8(卷积后的宽)] 压缩大小为1的维度
        x = torch.relu(x)  # 激活函数，尺寸不变
        x = torch.max_pool1d(x, x.size(2))  # 在每个通道做最大池化，[batch_size * 16(卷积层输出通道数) * 1]
        x = x.squeeze(2)  # 压缩维度2，[batch_size * 16(卷积层输出通道数)]
        x = self.dropout(x)  # dropout，尺寸不变
        logits = self.fc(x)  # 全连接输出头，[batch_size * n_class]
        return logits


dataset = MyDataset(max_len=10)  # 构造长度为10的句子输入，超过10的句子切掉，不足10的补<pad>
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
model = TextCNN(vocab_size=dataset.vocab_size, embed_size=128, n_class=dataset.n_class)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
for epoch in range(100):
    for feature, target in dataloader:
        optimizer.zero_grad()
        logits = model(feature)
        loss = criterion(logits, target.squeeze())
        loss.backward()
        optimizer.step()
    print('epoch:', epoch + 1, ', loss:', loss.item())


df = pd.read_csv('../../datasets/THUCNews/train.csv')
predict = model(feature).max(1)[1].tolist()
for i in range(len(feature.tolist())):
    print(' '.join([dataset.idx2token[idx] for idx in feature.tolist()[i]]),
          '---> label:',
          dict(zip(df['label'], df['class']))[predict[i]])
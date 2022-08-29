import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torchtext
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import re
from collections import Counter
from tqdm.notebook import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 能用gpu则用gpu


class MyDataset(Dataset):
    def __init__(self, file_path, tokenizer, stopwords, debug=True):
        df = pd.read_csv(file_path)
        df = df.dropna().reset_index(drop=True)
        if debug:
            df = df.sample(2000).reset_index(drop=True)
        counter = Counter()
        sentences = []
        for title in tqdm(df['title']):
            # 去除标点符号
            title = re.sub(r'[^\u4e00-\u9fa5]', '', title)
            tokens = [token for token in tokenizer(title.strip()) if token not in stopwords]
            counter.update(tokens)
            sentences.append(tokens)
        self.vocab = torchtext.vocab.vocab(counter, specials=['<unk>', '<pad>'])
        self.vocab.set_default_index(self.vocab['<unk>'])

        self.inputs = [self.vocab.lookup_indices(tokens) for tokens in sentences]
        self.labels = [[label] for label in df['label'].values.tolist()]
        self.n_class = len(df['label'].unique())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.LongTensor(self.inputs[idx]), torch.LongTensor(self.labels[idx])


file_path = '../data/THUCNews/train.csv'
tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='zh_core_web_sm')
stopwords = [line.strip() for line in open('../stopwords/cn_stopwords.txt', 'r', encoding='utf-8').readlines()]
dataset = MyDataset(file_path, tokenizer, stopwords, debug=False)


def collate_fn(batch_data):
    return pad_sequence([x for x, y in batch_data], padding_value=1), torch.tensor(
        [y for x, y in batch_data]).unsqueeze(1)


dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn)


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
        x = x.permute(1, 0)
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


model = TextCNN(vocab_size=len(dataset.vocab), embed_size=256, n_class=dataset.n_class).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
model.train()
for epoch in range(20):
    for feature, target in dataloader:
        feature = feature.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        logits = model(feature)
        loss = criterion(logits, target.squeeze())
        loss.backward()
        optimizer.step()
    print('epoch:', epoch + 1, ', loss:', loss.item())

model.eval()
df_train = pd.read_csv('../data/THUCNews/train.csv')
df_test = pd.read_csv('../data/THUCNews/test.csv')
for i, row in df_test.sample(10).iterrows():
    title = row['title']
    actual = row['class']
    title = re.sub(r'[^\u4e00-\u9fa5]', '', title)
    tokens = [token for token in tokenizer(title.strip()) if token not in stopwords]
    inputs = dataset.vocab.lookup_indices(tokens)
    inputs = torch.LongTensor(inputs).unsqueeze(1).to(device)
    predict = model(inputs)
    predict_class = dict(zip(df_train['label'], df_train['class']))[predict.max(1)[1].item()]
    print(' '.join(tokens), '||| actual:', actual, ', predict:', predict_class)

import torch
import torchtext
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import re
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

        # 构造输入和输出，输入是每三个词，输出是这三个词的下一个词，也就是简单的n-gram语言模型（n=3）
        self.inputs = []
        self.labels = []
        for sen in sentences:
            for i in range(len(sen) - 3):
                self.inputs.append(self.vocab.lookup_indices(sen[i: i + 3]))
                self.labels.append([self.vocab[sen[i + 3]]])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 返回一个x和一个y
        return torch.LongTensor(self.inputs[idx]), torch.LongTensor(self.labels[idx])


file_path = '../data/THUCNews/train.csv'
debug = True
tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='zh_core_web_sm')
stopwords = [line.strip() for line in open('../stopwords/cn_stopwords.txt', 'r', encoding='utf-8').readlines()]
dataset = MyDataset(file_path, tokenizer, stopwords)
dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)


class NNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, n_step, n_hidden):
        super().__init__()
        self.embed_size = embed_size
        self.n_step = n_step
        # vocab size投影到到embed size的空间中
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 构造一个隐藏层，输入大小为 步长 * embed size，输入大小为n_hidden
        self.linear = nn.Linear(n_step * embed_size, n_hidden)
        # 将n_hidden投影回vocab size大小
        self.output = nn.Linear(n_hidden, vocab_size)

    def forward(self, X):
        X = self.embedding(X)
        X = X.view(-1, self.n_step * self.embed_size)
        X = self.linear(X)
        X = torch.tanh(X)
        y = self.output(X)
        return y


# 初始化模型
model = NNLM(vocab_size=len(dataset.vocab), embed_size=256, n_step=3, n_hidden=256)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 查看模型
print(model)

# 训练20个epoch
for epoch in range(20):
    for train_input, train_label in dataloader:
        output = model(train_input)
        loss = criterion(output, train_label.squeeze_())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch:', epoch + 1, 'loss =', '{:.6f}'.format(loss))

# 使用训练好的模型进行预测，train_input直接是上面代码中的，直接用
# 模型输出之后取argmax，再用idx2token转回单词，查看效果，可以看到效果还可以，有上下文关系
predict = model(train_input).data.max(1, keepdim=True)[1].squeeze_().tolist()
input_list = train_input.tolist()
for i in range(len(input_list)):
    print(dataset.vocab.get_itos()[input_list[i][0]] + ' ' +
          dataset.vocab.get_itos()[input_list[i][1]] + ' ' +
          dataset.vocab.get_itos()[input_list[i][2]] + ' -> ' +
          dataset.vocab.get_itos()[predict[i]])

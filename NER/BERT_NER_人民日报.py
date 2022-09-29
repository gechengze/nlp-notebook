#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip

import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizer
from transformers import AutoModelForMaskedLM


# In[ ]:


class PD2014NER(Dataset):
    def __init__(self, source_path, target_path, bio2idx, tokenizer, max_len=512):
        super(Dataset, self).__init__()
        
        sources = open(source_path, 'r').readlines()  # 原始句子
        targets = open(target_path, 'r').readlines()  # BIO类别

        self.sentences = []
        self.labels = []
        
        for sentence, sentence_bio in tzip(sources, targets):
            if not sentence.strip() or len(sentence) > max_len - 2:
                continue
            self.sentences.append(tokenizer.encode(sentence.strip().split(' ')))
            self.labels.append([bio2idx[bio] for bio in sentence_bio.strip().split(' ')])
            
    def __getitem__(self, idx):
        return (torch.LongTensor(self.sentences[idx]), torch.LongTensor(self.labels[idx]))
    
    def __len__(self):
        return len(self.labels)


# In[ ]:


source_path = '../../datasets/NER/pd2014/source_BIO_2014_cropus.txt'
target_path = '../../datasets/NER/pd2014/target_BIO_2014_cropus.txt'

BIO = ['O', 'B_LOC', 'I_LOC', 'B_ORG', 'I_ORG', 'B_PER', 'I_PER', 'B_T', 'I_T']
bio2idx = {v: k for k, v in enumerate(BIO)}
idx2bio = {k: v for k, v in enumerate(BIO)}

tokenizer = BertTokenizer.from_pretrained('../../models/bert-base-chinese/')

dataset = PD2014NER(source_path, target_path, bio2idx, tokenizer)


# In[ ]:


num_class = len(BIO)

model = AutoModelForMaskedLM.from_pretrained('../../models/bert-base-chinese/')
model.cls.predictions.decoder = torch.nn.Linear(768, num_class, bias=True)
model = model.to(device)


# In[ ]:


def collate_fn(data_batch):
    x_batch, y_batch = [], []
    for x, y in data_batch:
        x_batch.append(x)
        y_batch.append(y)
    x_batch = pad_sequence(x_batch, padding_value=tokenizer.pad_token_id, batch_first=True)
    y_batch = pad_sequence(y_batch, padding_value=0, batch_first=True)
    return x_batch, y_batch

dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(20):
    total_loss_train = 0
    for x, y in tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)
        
        logits = model(x).logits
        logits = logits[:, 1: y.shape[1] + 1, :]  # 首尾的[CLS]和[SEP]去掉
        loss = criterion(logits.reshape(-1, num_class), y.reshape(-1))
        
        model.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_train += loss.item()
        
    print(f'Epochs:{epoch + 1}|Train Loss:{total_loss_train / len(dataset): .4f}')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import math
import pandas as pd
import random
import os

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import BertTokenizer
from tqdm import tqdm


# ### BERT输入数据构造

# In[ ]:


class MyDateset(Dataset):
    def __init__(self, tokenizer, dateset_path, dateset_type='train', num_sample=1000, max_len=128):
        super(MyDateset, self).__init__()

        self.tokenizer = tokenizer
        self.max_len = max_len

        df = pd.read_csv(os.path.join(dateset_path, dateset_type + '.csv')).sample(num_sample)

        paragraphs = []
        for c, f in zip(df['class'], df['file']):
            with open(os.path.join(dateset_path, c, f)) as file:
                paragraphs.append([sentence for paragraph in file.readlines()
                                   for sentence in paragraph.split('。') if sentence.strip()])

        self.examples = []

        for paragraph in tqdm(paragraphs):
            for i in range(len(paragraph) - 1):
                sentence_a = paragraph[i]
                # 50%的概率将连续两个句子拼接在一起，50%概率将不相邻的两个句子拼接在一起
                if random.random() < 0.5:
                    is_next = 1
                    sentence_b = paragraph[i + 1]
                else:
                    sentence_b = random.choice(random.choice(paragraphs))
                    is_next = 0

                # 将两个句子进行编码
                encoded = tokenizer.encode_plus(sentence_a, sentence_b,
                                                max_length=max_len, padding='max_length')

                # 如果两个句子拼起来超过最大长度，则跳过
                if len(encoded['input_ids']) > self.max_len:
                    continue

                encoded['is_next'] = is_next  # 是否相邻句子标识
                encoded = self.get_mlm_data(encoded)  # 进行掩码操作
                self.examples.append(encoded)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.examples[idx].input_ids),
            torch.LongTensor(self.examples[idx].token_type_ids),
            torch.LongTensor(self.examples[idx].attention_mask),
            torch.LongTensor([self.examples[idx].is_next]),
            torch.LongTensor(self.examples[idx].pred_positions),
            torch.LongTensor(self.examples[idx].pred_labels)
        )

    def get_mlm_data(self, encoded):
        candidate_pred_positions = []  # 除去特殊token外的所有token的位置
        for i, token in enumerate(encoded['input_ids']):
            # <CLS> <SEP> <PAD> 这三个token不做预测
            if token in [self.tokenizer.cls_token_id,
                         self.tokenizer.sep_token_id,
                         self.tokenizer.pad_token_id]:
                continue
            candidate_pred_positions.append(i)

        # 随机替换15%的token为<MASK>
        num_mlm_preds = max(1, round(sum(encoded['attention_mask']) * 0.15))

        # 要预测的token的位置
        pred_positions = sorted(random.sample(candidate_pred_positions, num_mlm_preds))

        # 要预测的token的真实值
        pred_labels = [encoded['input_ids'][pos] for pos in pred_positions]

        for pos in pred_positions:
            if random.random() < 0.8:
                # 80%的概率将token替换为<MASK>
                encoded['input_ids'][pos] = tokenizer.mask_token_id
            else:

                if random.random() < 0.5:
                    # 10%的概率token不变
                    continue
                else:
                    # 10%的概率随机替换成另外一个token
                    encoded['input_ids'][pos] = random.choice(range(106, tokenizer.vocab_size))

        # 将要预测的token位置和真实值pad到max_len * 0.15的长度，方便批量计算
        max_num_mlm_preds = round(self.max_len * 0.15)
        pred_positions += [0] * (max_num_mlm_preds - num_mlm_preds)
        pred_labels += [0] * (max_num_mlm_preds - num_mlm_preds)

        encoded['pred_positions'] = pred_positions
        encoded['pred_labels'] = pred_labels

        return encoded


# ### BERT模型

# #### Embdding层

# In[ ]:


class Embedding(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size):
        super(Embedding, self).__init__()
        # token embedding
        self.tok_embed = nn.Embedding(vocab_size, hidden_size)
        # 两个句子的embedding
        self.seg_embed = nn.Embedding(2, hidden_size)
        # 位置embedding
        self.pos_embed = nn.Embedding(max_len, hidden_size)
        # 层归一化
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, seg):
        # x输入：批量大小 * 步长
        seq_len = x.shape[1]
        # 位置编码，扩展维度后和x输入一样，批量大小 * 步长
        pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand_as(x)
        pos = pos.to(device)
        # 三个embedding相加
        embedded = self.tok_embed(x) + self.seg_embed(seg) + self.pos_embed(pos)
        return self.norm(embedded)


# #### 注意力层

# In[ ]:


# 缩放点积注意力
class ScaledDotProductionAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductionAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, attn_mask):
        # query/key/value：批量大小 * 头数 * 步长 * 向量维度
        # attn_mask尺寸：批量大小 * 头数 * 步长 * 步长
        d_k = key.shape[-1]  # key的维度
        # Q * K转置 / 根号d_k，scores尺寸：批量大小 * 头数 * 步长 * 步长
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn_weights = self.softmax(scores)
        return torch.matmul(attn_weights, value)  # 返回的结果尺寸：批量大小 * 头数 * 步长 * 向量维度


# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size  # 输入和输出的维度
        self.num_heads = num_heads  # 头数
        self.key_size = self.value_size = self.hidden_size // self.num_heads  # 输入输出维度必须可以整除头数，方便并行
        self.attention = ScaledDotProductionAttention()  # 缩放点积注意力
        self.W_Q = nn.Linear(hidden_size, hidden_size)  # Q的投影参数
        self.W_K = nn.Linear(hidden_size, hidden_size)  # K的投影参数
        self.W_V = nn.Linear(hidden_size, hidden_size)  # V的投影参数
        self.fc = nn.Linear(hidden_size, hidden_size)  # 全连接层，多头分开做自注意力后，再拼接起来，接一个全连接层

    def forward(self, query, key, value, attn_mask):
        # Q K V输入尺寸：批量大小 * 步长 * 维度
        # mask输入尺寸：批量大小 * 步长 * 步长
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        # 方便多头并行计算，QKV投影后reshape成 批量大小 * 步长 * 头数 * 维度，再交换1、2维度，变成 批量大小 * 头数 * 步长 * 维度
        q_s = self.W_Q(query).reshape(batch_size, -1, self.num_heads, self.key_size).transpose(1, 2)
        k_s = self.W_Q(key).reshape(batch_size, -1, self.num_heads, self.key_size).transpose(1, 2)
        v_s = self.W_Q(value).reshape(batch_size, -1, self.num_heads, self.value_size).transpose(1, 2)

        # mask处理成 批量大小 * 头数 * 步长 * 步长
        attn_mask = attn_mask.data.eq(0).unsqueeze(1).expand(batch_size, seq_len, seq_len)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        # context尺寸：批量大小 * 头数 * 步长 * 单个头的维度
        context = self.attention(q_s, k_s, v_s, attn_mask)

        # context尺寸：批量大小 * 步长 * hidden_size
        context = context.transpose(1, 2).reshape(batch_size, -1, self.hidden_size)

        output = self.fc(context)

        return output


# #### 前馈网络

# In[ ]:


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionWiseFFN(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        # 两个全连接层，使用gelu作为激活函数
        super(PositionWiseFFN, self).__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_size)
        self.fc2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


# #### 残差连接和层归一化

# In[ ]:


class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        # 层归一化 + 残差连接
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(norm_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        return self.layer_norm(x + self.dropout(y))


# #### Transformer Encoder块

# In[ ]:


class EncoderBlock(nn.Module):
    def __init__(self, num_heads, hidden_size, ffn_size, dropout):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.add_norm1 = AddNorm(norm_shape=hidden_size, dropout=dropout)
        self.ffn = PositionWiseFFN(hidden_size=hidden_size, ffn_size=ffn_size)
        self.add_norm2 = AddNorm(norm_shape=hidden_size, dropout=dropout)

    def forward(self, x, attn_mask):
        output = self.add_norm1(x, self.attention(x, x, x, attn_mask))
        output = self.add_norm2(output, self.ffn(output))
        return output


# #### BERT模型，是否相邻句子+MASK位置预测

# In[ ]:


class BERT(nn.Module):
    def __init__(self, num_layers, num_heads, vocab_size, max_len, hidden_size, ffn_size, dropout):
        super(BERT, self).__init__()

        self.embedding = Embedding(vocab_size=vocab_size, max_len=max_len, hidden_size=hidden_size)

        self.layers = nn.Sequential()

        for i in range(num_layers):
            self.layers.add_module(f'{i}', EncoderBlock(num_heads=num_heads, hidden_size=hidden_size,
                                                        ffn_size=ffn_size, dropout=dropout))

        # 是否下一个句子预测
        self.is_next_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(hidden_size, 2)
        )

        # 预测mask掉的token
        self.mask_lm = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(self, input_ids, segment_ids, attn_mask, pred_positions):
        output = self.embedding(input_ids, segment_ids)

        for layer in self.layers:
            output = layer(output, attn_mask)

        # 用输出的第一个位置，即[CLS]的768维的向量表示，拿来做是否是相邻句子的预测
        cls_output = output[:, 0]
        logit_is_next = self.is_next_classifier(cls_output)

        # [MASK]位置的预测
        batch_size, num_pred_positions = pred_positions.shape
        pred_positions = pred_positions.reshape(-1)
        
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        batch_idx = batch_idx.to(device)
        
        mask_output = output[batch_idx, pred_positions]
        mask_output = mask_output.reshape((batch_size, num_pred_positions, -1))
        logit_mask = self.mask_lm(mask_output)
        
        return logit_is_next, logit_mask


# ### 训练模型

# In[ ]:


batch_size = 64
num_layers = 12
num_heads = 12
max_len = 128
hidden_size = 768
ffn_size = 768
dropout = 0.2
lr = 1e-4

tokenizer = BertTokenizer.from_pretrained('../../models/bert-base-chinese/')
vocab_size = tokenizer.vocab_size


# In[ ]:


dateset_path = '../../datasets/THUCNews'
dataset_type = 'train'

train_dataset = MyDateset(tokenizer, dateset_path, dataset_type, num_sample=1000)
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# In[ ]:


model = BERT(num_layers=num_layers, num_heads=num_heads, vocab_size=vocab_size,
             max_len=max_len, hidden_size=hidden_size, ffn_size=ffn_size, dropout=dropout)
model = model.to(device)

# 模型可学习参数量
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# In[ ]:


for epoch in range(20):
    total_loss_train = 0
    for input_ids, segment_ids, attn_mask, is_next, pred_positions, pred_labels in data_loader:

        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        attn_mask = attn_mask.to(device)
        is_next = is_next.to(device)
        pred_positions = pred_positions.to(device)
        pred_labels = pred_labels.to(device)

        logit_is_next, logit_mask = model(input_ids, segment_ids, attn_mask, pred_positions)

        loss_is_next = criterion(logit_is_next, is_next.view(-1))

        loss_mask = criterion(logit_mask.view(-1, vocab_size), pred_labels.view(-1))

        loss = loss_is_next + loss_mask
        total_loss_train += loss.item()
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epochs:{epoch + 1}|Train Loss:{total_loss_train / len(train_dataset): .4f}')




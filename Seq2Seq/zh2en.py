import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pandas as pd
from collections import Counter
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_csv('cmn.txt', sep='\t', header=None, names=['en', 'zh'])
my_vocab = {}
counter = Counter()
for string_ in df['zh']:
    counter.update(list(string_))
my_vocab['zh'] = vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
my_vocab['zh'].set_default_index(my_vocab['zh']['<unk>'])

counter = Counter()
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
for string_ in df['en']:
    counter.update(en_tokenizer(string_))
my_vocab['en'] = vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
my_vocab['en'].set_default_index(my_vocab['en']['<unk>'])


def data_process(df):
    data = []
    for raw_zh, raw_en in zip(df['zh'], df['en']):
        zh_tensor_ = torch.tensor([my_vocab['zh'][token] for token in list(raw_zh)],
                                  dtype=torch.long)
        en_tensor_ = torch.tensor([my_vocab['en'][token] for token in en_tokenizer(raw_en)],
                                  dtype=torch.long)
        data.append((zh_tensor_, en_tensor_))
    return data


train_data = data_process(df)

BATCH_SIZE = 256
PAD_IDX = my_vocab['zh']['<pad>']
BOS_IDX = my_vocab['zh']['<bos>']
EOS_IDX = my_vocab['zh']['<eos>']


def generate_batch(data_batch):
    zh_batch, en_batch = [], []
    for zh_item, en_item in data_batch:
        zh_batch.append(torch.cat([torch.tensor([BOS_IDX]), zh_item, torch.tensor([EOS_IDX])], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    zh_batch = pad_sequence(zh_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return zh_batch, en_batch


train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embed(x))
        enc_output, enc_hidden = self.rnn(embedded)
        return enc_output, enc_hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, hidden):
        embedded = self.dropout(self.embed(y))
        dec_output, hidden = self.rnn(embedded, hidden)
        dec_output = self.fc(dec_output)
        return dec_output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        enc_output, hidden = self.encoder(src)
        max_len, batch_size = tgt.shape[0], tgt.shape[1]
        output = torch.zeros(max_len, batch_size, self.decoder.vocab_size).to(device)
        y = tgt[0, :]
        for t in range(1, max_len):
            y.unsqueeze_(0)
            y, hidden = self.decoder(y, hidden)
            y.squeeze_(0)
            output[t] = y
            y = y.max(1)[1]
        return output


enc = Encoder(vocab_size=len(my_vocab['zh']), embed_size=64, hidden_size=64)
dec = Decoder(vocab_size=len(my_vocab['en']), embed_size=64, hidden_size=64)
model = Seq2Seq(enc, dec, device).to(device)


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX).to(device)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

model.train()
for epoch in range(100):
    epoch_loss = 0
    for src, tgt in train_iter:
        src = src.to(device)
        tgt = tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt)
        output = output[1:].view(-1, output.shape[-1])
        tgt = tgt[1:].view(-1)
        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        epoch_loss += loss.item()
    print('epoch:', epoch + 1, ', loss:', epoch_loss / len(train_iter))

model.eval()


def translate(zh, max_len=10):
    zh_idx = [my_vocab['zh']['<bos>']] + my_vocab['zh'].lookup_indices(list(zh)) + [my_vocab['zh']['<eos>']]
    zh_idx = torch.tensor(zh_idx, dtype=torch.long, device=device).unsqueeze_(1)
    en_bos = my_vocab['en']['<bos>']
    enc_output, hidden = model.encoder(zh_idx)
    preds = []
    y = torch.tensor([en_bos], dtype=torch.long, device=device)
    for t in range(max_len):
        y.unsqueeze_(1)
        y, hidden = model.decoder(y, hidden)
        y.squeeze_(1)
        y = y.max(1)[1]
        if y.item() == my_vocab['en']['<eos>']:
            break
        preds.append(my_vocab['en'].get_itos()[y.item()])
    return ' '.join(preds)


print(translate('我是一个学生'))
for zh in df['zh'][0: 100]:
    print(zh, '   ==>   ', translate(zh, max_len=10))
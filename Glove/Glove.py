import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torchtext
import jieba
import re
import pandas as pd
from tqdm.notebook import tqdm

df = pd.read_csv('../data/THUCNews/train.csv').dropna().sample(100000)
stopwords = [line.strip() for line in open('../stopwords/cn_stopwords.txt', 'r', encoding='utf-8').readlines()]

f = open('./stanford-Glove/THUCNews.txt', 'w')
for title in tqdm(df['title']):
    # 去除标点符号
    title = re.sub(r'[^\u4e00-\u9fa5]', '', title)
    tokens = [token for token in jieba.cut(title.strip()) if token not in stopwords]
    f.write(' '.join(tokens) + '\n')
f.close()

# cd stanford-Glove
# make
# sh demo.sh

# 加载Glove预训练的词向量
embeddings = torchtext.vocab.Vectors(name='chinese-glove-vectors.txt')


# 简单计算几个词之间的余弦相似度
def cal_cos(token1, token2):
    cos = torch.nn.CosineSimilarity(dim=0)
    return cos(embeddings.get_vecs_by_tokens(token1), embeddings.get_vecs_by_tokens(token2)).item()


print(cal_cos('中国', '美国'))
print(cal_cos('中国', '韩国'))
print(cal_cos('中国', '第一'))
print(cal_cos('图', '组图'))
print(cal_cos('巴萨', '皇马'))
print(cal_cos('巴萨', '中国'))

# 通过Glove生存的vovab.txt创建torchtext.vocab.vocab
df_vocab = pd.read_csv('stanford-Glove/vocab.txt', sep=' ', header=None, names=['token', 'cnt'])
vocab = torchtext.vocab.vocab(dict(zip(df_vocab['token'], df_vocab['cnt'])))

# 将预训练vectors加载到Embdding网络中
# freeze为True，则冻结embed层的参数
embed = torch.nn.Embedding.from_pretrained(embeddings.vectors, freeze=True)
print(embed.weight)
print(embed.weight.requires_grad)

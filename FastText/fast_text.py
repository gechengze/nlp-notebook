import fasttext
import pandas as pd
import re
import csv
from torchtext.data.utils import get_tokenizer


def get_fasttext_data(df):
    df = df.dropna().sample(20000).reset_index(drop=True)
    df['label'] = '__label__' + df['class']

    tokenizer = get_tokenizer('spacy', language='zh_core_web_sm')
    stopwords = [line.strip() for line in open('../stopwords/cn_stopwords.txt', 'r',
                                               encoding='utf-8').readlines()]

    def process_text(title):
        title = re.sub(r'[^\u4e00-\u9fa5]', '', title)  # 去除标点符号
        tokens = [token for token in tokenizer(title.strip()) if token not in stopwords]
        return ' '.join(tokens)

    df['text'] = df['title'].map(process_text)
    df['label_text'] = df['label'] + ',' + df['text']
    return df[['label_text']]


fasttext_train = get_fasttext_data(pd.read_csv('../data/THUCNews/train.csv'))
fasttext_train.to_csv('./fasttext_train.txt', header=None, index=False,
                      quoting=csv.QUOTE_NONE, escapechar=' ')

fasttext_test = get_fasttext_data(pd.read_csv('../data/THUCNews/test.csv'))
fasttext_test.to_csv('./fasttext_test.txt', header=None, index=False,
                     quoting=csv.QUOTE_NONE, escapechar=' ')
# 训练模型
model = fasttext.train_supervised('./fasttext_train.txt')
print(model.predict('盘点 明星 私生子 爱情 事故 戏外 戏图'))
print(model.predict('微软 员工 微博 泄密 手机 遭 解雇'))
# 训练集的precision和recall
print(model.test('fasttext_train.txt'))
# 测试集的precision和recall
print(model.test('fasttext_test.txt'))

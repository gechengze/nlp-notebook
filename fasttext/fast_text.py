import fasttext
import pandas as pd
import jieba
import re
import csv

stopwords = [line.strip() for line in open('../../stopwords/cn_stopwords.txt', 'r', encoding='utf-8').readlines()]


def process_text(title):
    # 去除标点符号
    title = re.sub(r'[^\u4e00-\u9fa5]', '', title)
    # jieba分词
    sentence_seged = jieba.cut(title.strip())
    outstr = ''
    for word in sentence_seged:
        if word != '\t' and word not in stopwords:
            outstr += word
            outstr += ' '
    return outstr


df_train = pd.read_csv('../../datasets/THUCNews/train.csv')
df_train = df_train.dropna().sample(200000).reset_index(drop=True)
# 处理成fasttext所需的格式
df_train['label'] = '__label__' + df_train['class']
df_train['text'] = df_train['title'].map(process_text)
df_train['label_text'] = df_train['label'] + ',' + df_train['text']
fasttext_train = df_train[['label_text']]
fasttext_train.to_csv('./fasttext_train.txt', header=None, index=False, quoting=csv.QUOTE_NONE, escapechar=' ')

df_test = pd.read_csv('../../datasets/THUCNews/test.csv')
df_test['label'] = '__label__' + df_test['class']
df_test['text'] = df_test['title'].map(process_text)
df_test['label_text'] = df_test['label'] + ',' + df_test['text']
fasttext_test = df_test[['label_text']]
fasttext_test.to_csv('./fasttext_test.txt', header=None, index=False, quoting=csv.QUOTE_NONE, escapechar=' ')

# 训练模型
model = fasttext.train_supervised('./fasttext_train.txt')

print(model.predict('盘点 明星 私生子 爱情 事故 戏外 戏图'))
print(model.predict('微软 员工 微博 泄密 手机 遭 解雇'))

# 训练集的precision和recall
print(model.test('fasttext_train.txt'))

# 测试集的precision和recall
print(model.test('fasttext_test.txt'))

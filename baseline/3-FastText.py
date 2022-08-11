# !wget https://github.com/facebookresearch/fastText/archive/0.2.0.zip
# !unzip 0.2.0.zip
# %cd fastText-0.2.0
# !make

# 1 is positive, 0 is negative
f = open('train.txt', 'w')
f.write('__label__1 i love you\n')
f.write('__label__1 he loves me\n')
f.write('__label__1 she likes baseball\n')
f.write('__label__0 i hate you\n')
f.write('__label__0 sorry for that\n')
f.write('__label__0 this is awful')
f.close()

f = open('test.txt', 'w')
f.write('sorry hate you')
f.close()

# !./fasttext supervised -input train.txt -output model -dim 2

# !cat test.txt
# !./fasttext predict model.bin test.txt

import torch
import torch.nn as nn
import torch.optim as optim


class NNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, n_step, n_hidden):
        super().__init__()
        self.embed_size = embed_size
        self.n_step = n_step
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(n_step * embed_size, n_hidden)
        self.output = nn.Linear(n_hidden, vocab_size)

    def forward(self, X):
        X = self.embedding(X)
        X = X.view(-1, self.n_step * self.embed_size)
        X = self.linear(X)
        X = torch.tanh(X)
        y = self.output(X)
        return y


if __name__ == '__main__':
    sentences = ['i like dog', 'i love coffee', 'i hate milk']
    token_list = list(set(' '.join(sentences).split()))
    token2idx = {token: i for i, token in enumerate(token_list)}
    idx2token = {i: token for i, token in enumerate(token_list)}

    model = NNLM(vocab_size=len(token_list), embed_size=2, n_step=2, n_hidden=8)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch = []
    target_batch = []
    for sen in sentences:
        input_batch.append([token2idx[n] for n in sen.split()[:-1]])
        target_batch.append(token2idx[sen.split()[-1]])
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # Training
    for epoch in range(10000):
        optimizer.zero_grad()
        output = model(input_batch)

        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # Predict
    predict = model(input_batch).data.max(1, keepdim=True)[1]

    # Test
    print([sen.split()[:2] for sen in sentences], '->', [idx2token[n.item()] for n in predict.squeeze()])

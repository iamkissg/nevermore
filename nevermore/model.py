# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable


class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, rnn_type="LSTM"):
        """

        :param vocab_size: size of vocabulary
        :param embedding_dim: dimension of embed vector
        :param hidden_size: hidden size of RNN. M -> M RNN
        :param num_layers: number of layers of RNN
        :param rnn_type: default is "LSTM", others can be "GRU" or vallina RNN
        """
        super(CharRNN, self).__init__()
        self.rnn_type = rnn_type

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.Embedding(vocab_size, embedding_dim=embedding_dim)

        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, self.hidden_size, num_layers, batch_first=True, dropout=0.5)
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, self.hidden_size, num_layers, batch_first=True, dropout=0.5)
        else:
            self.rnn = nn.RNN(embedding_dim, self.hidden_size, num_layers, batch_first=True, dropout=0.5)

        self.decoder = nn.Linear(self.hidden_size, vocab_size)
        self.relu = nn.ReLU()

    def forward(self, inputs, hidden):
        """

        :param inputs: the size of inputs is batch_size x string_length
        :param hidden: the size of hidden is (num_layers * num_directions) x batch_size x hidden_size
        :return:
        """
        embedded = self.encoder(inputs)  # the size of embedded is batch_size x seq_length x embedding_dim
        outputs, hidden = self.rnn(embedded, hidden)
        # Reshape outputs from (batch, seq_len, hidden_size) to (batch x seq_len, hidden_size)
        outputs = outputs.contiguous().view(-1, self.hidden_size)
        outputs = self.decoder(outputs)  # size: batch x vocab_size)
        # outputs = self.relu(outputs)  # why relu?

        return outputs, hidden

    def init_hidden(self, batch_size):
        if self.rnn_type == "LSTM":
            # LSTM has 2 states, one for cell state, another one for hidden state
            return (Variable(torch.randn(self.num_layers * 1, batch_size, self.hidden_size)),
                    Variable(torch.randn(self.num_layers * 1, batch_size, self.hidden_size)))
        else:
            return Variable(torch.randn(self.num_layers * 1, batch_size, self.hidden_size))


if __name__ == '__main__':
    print("TOY!!!JUST FOR TEST!!!\n")

    import numpy as np
    import torch.optim as optim
    import matplotlib.pyplot as plt

    # Data and prepare
    sentences = "少春眠不觉晓处处闻啼鸟夜来风雨声花落知多少"
    vocab = {w for s in sentences for w in s}
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    isentences = np.array([word2idx[w] for s in sentences for w in s])
    inputs = torch.LongTensor(isentences[:isentences.shape[0] - 1]).view(-1, 1)
    results = torch.LongTensor(isentences[1:]).view(-1, 1)

    batch_size = 1
    epochs = 1000  # 经过长时间的训练, 程序能够"记"住一些信息

    # optimizer parameters
    lr = 0.01
    lr_decay = 0.00001
    betas = (0.9, 0.999)

    print(sentences)
    print(word2idx)
    print(isentences)
    print(inputs)
    print(results)
    print("#" + "=" * 79 + "\n")

    char_rnn = CharRNN(len(vocab), embedding_dim=5, hidden_size=5, num_layers=1, rnn_type="LSTM")

    # Freeze parameters of Embedding Layer
    for param in char_rnn.encoder.parameters():
        param.requires_grad = False
    parameters = filter(lambda p: p.requires_grad, char_rnn.parameters())

    optimizer = optim.Adam(params=parameters, lr=lr, weight_decay=lr_decay, betas=betas)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()

        hidden = char_rnn.init_hidden(batch_size)  # every epoch, we need to re-initial the hidden state
        outputs, hidden = char_rnn.forward(Variable(inputs), hidden)
        loss = criterion(outputs, Variable(results).contiguous().view(-1))
        losses.append(loss.data.numpy()[0])
        loss.backward()
        optimizer.step()

    hidden = char_rnn.init_hidden(batch_size)  # every epoch, we need to re-initial the hidden state
    a, h = char_rnn.forward((Variable(inputs)), hidden)
    print("indexes", np.argmax(a.data.numpy(), axis=1))
    print([idx2word[i] for i in np.argmax(a.data.numpy(), axis=1)])
    plt.plot(np.arange(len(losses)), np.array(losses))
    plt.show()

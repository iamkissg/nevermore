# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import config
from util import load_embedding, get_weight_matrix, load_vocab, load_word2idx_idx2word


class PoetryGenerator(nn.Module):
    """
    CharRNN based Poetry Generator
    """

    def __init__(self, vocab_size, embedding_dim, rnn_hidden_size, rnn_num_layers=1, rnn_type="LSTM",
                 tie_weights=False):
        super(PoetryGenerator, self).__init__()

        self.rnn_type = rnn_type
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers

        self.encoder = self.init_embedding(embedding_dim)  # use embedding layer as first layer

        # TODO: better CSM
        # ==============================================================================================
        # Convolutional Sentence Model (CSM) layers, compresses a line (sequence of vectors) to a vector
        # use full convolutional layers without pooling
        # ==============================================================================================
        self.csm_l1 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(1, 2, 1), stride=(1, 1, 1)),
            nn.Dropout2d())
        self.csm_l2 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(1, 2, 1), stride=(1, 1, 1)),
            nn.Dropout2d())
        self.csm_l3 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(1, 3, 1), stride=(1, 1, 1)),
            nn.Dropout2d())
        self.csm_l4 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(1, 3, 1), stride=(1, 1, 1)),
            nn.Dropout2d())

        # TODO: better context
        # ====================================================================================================
        # Context Model (CM) layers, compresses vectors of lines to one vector
        # for convenience, define 2 selectable layers for the training data is QUATRAIN which contains 4 lines
        # ====================================================================================================
        # Compress 2 lines context into 1 vector
        self.cm_21 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(2, 1), stride=(1, 1)),
            nn.Dropout2d())
        # Compress 3 lines context into 1 vector
        self.cm_31 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3, 1), stride=(1, 1)),
            nn.Dropout2d())

        # ==============================================================================================
        # Recurrent Generation Model (RGM) layers,
        # generates one word according to the previous words in the current line and the previous lines
        # ==============================================================================================
        # the inputs is concatenation of word embedding and lines vector (the same dimension as word embedding just now)
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(self.rnn_hidden_size * 2, self.rnn_hidden_size, self.rnn_num_layers, batch_first=True,
                               dropout=0.5)
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(self.rnn_hidden_size * 2, self.rnn_hidden_size, self.rnn_num_layers, batch_first=True,
                              dropout=0.5)
        else:
            self.rnn = nn.RNN(self.rnn_hidden_size * 2, self.rnn_hidden_size, self.rnn_num_layers, batch_first=True,
                              dropout=0.5)

        self.decoder = nn.Linear(self.rnn_hidden_size, vocab_size)

        # tie weights, e.i. use same weight as encoder for decoder, (I learned this trick from PyTorch Example).
        if tie_weights:
            self.decoder.weight = self.encoder.weight

    def forward(self, w, s, hidden):
        # The code below is designed for mini-batch training

        # word embedding and reshape to (batch size, seq lenth, hidden size)
        w = self.encoder(w).view(w.size(0), -1, self.rnn_hidden_size)  # for 7-character quatrain, seq_length is 6
        # embedding requires indices.dim() < 2, so reshape s to (batch size, num_lines * num_words_each_line)
        # then embed words in sentences into embedded space, and then reshape to
        # (batch size, num_channel(used in cnn), depth(num_ines), height(num_words_each_line, hidden_size)
        s = self.encoder(s.contiguous().view(w.size(0), -1)).contiguous().view(
            w.size(0), 1, s.size(-2), s.size(-1), self.rnn_hidden_size)

        # ===
        # CSM
        # ===
        # s - (batch size, channel size, depth, height, width)
        # after compressing, v - (batch size, channel size, depth, 1, width)
        v = F.leaky_relu(self.csm_l1(s))  # TODO: more helpful activation function
        v = F.leaky_relu(self.csm_l2(v))
        v = F.leaky_relu(self.csm_l3(v))
        if s.size(3) == 7:
            v = F.leaky_relu(self.csm_l4(v))
        assert v.size(3) == 1

        # ==
        # CM
        # ==
        # reshape v into 4 dimensions tensor (remove height)
        v = v.view(w.size(0), 1, s.size(2), self.rnn_hidden_size)  # (batch size, channel size, depth, width)
        if s.size(2) > 1:
            cm = self.cm_21 if s.size(2) == 2 else self.cm_31  # 3 lines to 1 vector or 2 lines to vector
            u = F.leaky_relu(cm(v))
        else:
            u = v  # if generating 2nd line, there is only 1 previous line, no need compression.
        u = u.view(w.size(0), 1, self.rnn_hidden_size)  # reshape to (batch, channel size, hidden size), remove depth
        u = u.repeat(1, w.size(1), 1)  # for convenience again. (Forgive me - -.)

        # ===
        # RGM
        # ===
        # The input of RGM is the concatenation of word embedding vector and sentence context vector
        uw = torch.cat([w, u], dim=2)
        uw = uw.view(w.size(0), w.size(1), self.rnn_hidden_size * 2)  # reshape to (batch size, seq length, hidden size)
        y, hidden = self.rnn(uw, hidden)

        y = y.contiguous().view(-1, self.rnn_hidden_size)
        y = self.decoder(y)

        return y, hidden

    def init_hidden(self, batch_size):
        # Zero initial
        # (other initialization is ok, I haven't tried others.)
        if self.rnn_type == "LSTM":
            # LSTM has 2 states, one for cell state, another one for hidden state
            return (Variable(torch.zeros(self.rnn_num_layers * 1, batch_size, self.rnn_hidden_size)),
                    Variable(torch.zeros(self.rnn_num_layers * 1, batch_size, self.rnn_hidden_size)))
        else:
            return Variable(torch.zeros(self.rnn_num_layers * 1, batch_size, self.rnn_hidden_size))

    def init_embedding(self, embedding_dim):
        # use `pre-trained` embedding layer, this trick is learned from machinelearningmastery.com
        vocab = load_vocab()
        word2idx = {w: i for i, w in enumerate(vocab)}
        raw_embedding = load_embedding(config.path_embedding)
        embedding_weight = get_weight_matrix(raw_embedding, word2idx)

        embedding = nn.Embedding(len(vocab), embedding_dim=embedding_dim)
        embedding.weight = nn.Parameter(torch.from_numpy(embedding_weight).float())

        return embedding


if __name__ == '__main__':
    print("TOY EXAMPLE, JUST FOR TEST!!!")

    import numpy as np
    import torch.optim as optim

    vocab = load_vocab()
    word2idx, idx2word = load_word2idx_idx2word(vocab)
    poetry = "鹤 湖 东 去 水 茫 茫	一 面 风 泾 接 魏 塘	看 取 松 江 布 帆 至	鲈 鱼 切 玉 劝 郎 尝"
    sentences = [s.split() for s in poetry.split("\t")]
    isentences = [[word2idx[w] for w in s] for s in sentences]
    print(sentences)
    print(isentences)

    batch_size = 1
    epochs = 10  # 经过长时间的训练, 程序能够"记"住一些信息

    # optimizer parameters
    lr = 0.01
    decay_factor = 0.00001
    betas = (0.9, 0.999)

    pg = PoetryGenerator(vocab_size=len(vocab), embedding_dim=256, rnn_hidden_size=256, tie_weights=True,
                         rnn_num_layers=3)

    # Freeze parameters of Embedding Layer
    for param in pg.encoder.parameters():
        param.requires_grad = False
    parameters = filter(lambda p: p.requires_grad, pg.parameters())

    optimizer = optim.Adam(params=parameters, lr=lr, betas=betas)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 / decay_factor)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for iss, s in enumerate(isentences[1:], start=1):
            hidden = pg.init_hidden(batch_size)
            pg.zero_grad()
            y, hidden = pg.forward(w=Variable(torch.LongTensor([s[:-1]])),
                                   s=Variable(torch.LongTensor([isentences[:iss]])),
                                   hidden=hidden)

            loss = criterion(y, Variable(torch.LongTensor([s[1:]])).view(-1))
            print(loss.data.numpy())
            loss.backward()
            optimizer.step()

    hidden = pg.init_hidden(batch_size)  # every epoch, we need to re-initial the hidden state
    for i, w in enumerate("松江"):
        a, hidden = pg.forward(w=Variable(torch.LongTensor([word2idx[w]])),
                               s=Variable(torch.LongTensor(isentences[:1])),
                               hidden=hidden)
        print("indexes", np.argmax(a.data.numpy(), axis=1))
        print([idx2word[i] for i in np.argmax(a.data.numpy(), axis=1)])

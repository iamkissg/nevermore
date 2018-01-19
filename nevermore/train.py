# -*- coding: utf-8 -*-

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import Poemsets
from model import PoetryGenerator
import config
from util import load_word2idx_idx2word, load_vocab


vocab = load_vocab()
word2idx, idx2word = load_word2idx_idx2word(vocab)

# training hyper-parameters
epochs = 1000
batch_size = 1024
hidden_size = 256
lr = 0.01
decay_factor = 1.004
betas = (0.9, 0.999)

mode = "7"  # used for save checkpoint file
poemset = Poemsets(os.path.join(config.dir_rnnpg, "partitions_in_Table_2", "rnnpg", "qtrain_7"))
poemloader = DataLoader(poemset, batch_size=batch_size, shuffle=True)


# note embedding size is the same as rnn_hidden_size at this version
pg = PoetryGenerator(vocab_size=len(vocab), embedding_dim=256, rnn_hidden_size=hidden_size, rnn_num_layers=3, tie_weights=True)

# Freeze parameters of Embedding Layer
for params in pg.encoder.parameters():
    params.requires_grad = False
parameters = filter(lambda p: p.requires_grad, pg.parameters())

optimizer = optim.Adam(params=parameters, lr=lr, betas=betas, weight_decay=1e-7)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 / decay_factor)
criterion = nn.CrossEntropyLoss()

# ====================================================================
# uncomment following 5 lines to load checkpoint file
# and adjust learning rate, it's useful when training with checkpoint file
# ====================================================================
# checkpoint = torch.load(os.path.join(config.dir_chkpt, mode, "288_new.chkpt"))
# pg.load_state_dict(checkpoint)
# for i in range(289):
#     scheduler.step()
# for epoch in range(288, epochs):  # uncomment this line for avoiding overwriting exsiting checkpoint file

for epoch in range(epochs):
    scheduler.step()  # update learning rate

    losses = []
    for poems in poemloader:
        # train with line 2
        hidden = pg.init_hidden(len(poems))
        pg.zero_grad()
        y, hidden = pg.forward(w=Variable(torch.LongTensor(poems[:, 1, :-1])),
                               s=Variable(torch.LongTensor(poems[:, :1])),
                               hidden=hidden)
        loss = criterion(y, Variable(torch.LongTensor(poems[:, 1, 1:])).contiguous().view(-1))
        loss.backward()
        optimizer.step()

        # train with line 3
        hidden = pg.init_hidden(len(poems))
        pg.zero_grad()
        y, hidden = pg.forward(w=Variable(torch.LongTensor(poems[:, 2, :-1])),
                               s=Variable(torch.LongTensor(poems[:, :2])),
                               hidden=hidden)
        loss = criterion(y, Variable(torch.LongTensor(poems[:, 2, 1:])).contiguous().view(-1))
        loss.backward()
        optimizer.step()

        # train with line 4
        hidden = pg.init_hidden(len(poems))
        pg.zero_grad()
        y, hidden = pg.forward(w=Variable(torch.LongTensor(poems[:, 3, :-1])),
                               s=Variable(torch.LongTensor(poems[:, :3])),
                               hidden=hidden)
        loss = criterion(y, Variable(torch.LongTensor(poems[:, 3, 1:])).contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(loss.data[0])

        # the following lines is used for checking whether the network works well.
        hidden = pg.init_hidden(batch_size)
        _, hidden = pg.forward(w=Variable(torch.LongTensor([[word2idx[w] for w in list("一")]])),
                               s=Variable(torch.LongTensor([[word2idx[w] for w in list("两个黄鹂鸣翠柳")]])),
                               hidden=hidden)
        _, hidden = pg.forward(w=Variable(torch.LongTensor([[word2idx[w] for w in list("行")]])),
                               s=Variable(torch.LongTensor([[word2idx[w] for w in list("两个黄鹂鸣翠柳")]])),
                               hidden=hidden)
        a, hidden = pg.forward(w=Variable(torch.LongTensor([[word2idx[w] for w in list("白")]])),
                               s=Variable(torch.LongTensor([[word2idx[w] for w in list("两个黄鹂鸣翠柳")]])),
                               hidden=hidden)
        # print("indexes", np.argmax(a.data.numpy(), axis=1))
        # print([idx2word[i] for i in np.argmax(a.data.numpy(), axis=1)])
    torch.save(pg.state_dict(), "{dir}/{mode}/{epoch}_new2.chkpt".format(dir=config.dir_chkpt, mode=mode, epoch=epoch))

# -*- coding: utf-8 -*-

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import Poemsets
from model import CharRNN
import config

# training hyper-parameters
epochs = 1000
batch_size = 1024
embedding_dim = 256
hidden_size = 256

lr = 0.1
decay_factor = 1.00004
betas = (0.9, 0.999)

mode = "5"
poemset = Poemsets(os.path.join(config.dir_poemlm, "qts_{m}.txt".format(m=mode)))
poemloader = DataLoader(poemset, batch_size=batch_size, shuffle=True)

char_rnn = CharRNN(len(poemset.vocab), embedding_dim=embedding_dim, hidden_size=hidden_size, num_layers=3)
# checkpoint = torch.load(os.path.join(config.dir_chkpt, mode, "310.chkpt"))
# char_rnn.load_state_dict(checkpoint)

# Freeze parameters of Embedding Layer
# for param in char_rnn.encoder.parameters():
#     param.requires_grad = False
# parameters = filter(lambda p: p.requires_grad, char_rnn.parameters())

# optimizer = optim.Adam(params=parameters, lr=lr, betas=betas)
optimizer = optim.Adam(params=char_rnn.parameters(), lr=lr, betas=betas)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 / decay_factor)
criterion = nn.CrossEntropyLoss()

for ep in range(310, epochs + 1):
    scheduler.step()
    print("LR", optimizer.param_groups[0]["lr"])

    losses = []
    for poems in poemloader:
        optimizer.zero_grad()

        inputs = poems[:, :poems.size(1) - 1]
        targets = poems[:, 1:].contiguous().view(-1)
        inputv = Variable(inputs)
        targetv = Variable(targets)

        hidden = char_rnn.init_hidden(poems.size(0))
        outputs, hidden = char_rnn.forward(inputv, hidden)
        loss = criterion(outputs, targetv)
        print(loss.data)
        loss.backward()
        losses.append(loss)

        optimizer.step()

        print([poemset.idx2word[i] for i in poems.numpy()[0]])
        print(["  "] + [poemset.idx2word[i] for i in np.argmax(outputs.data.numpy()[:int(mode) - 1], axis=1)])

    torch.save(char_rnn.state_dict(), "{dir}/{mode}/{epoch}.chkpt".format(dir=config.dir_chkpt, mode=mode, epoch=ep))

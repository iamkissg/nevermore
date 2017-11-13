# -*- coding: utf-8 -*-

import os
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import config
from model import CharRNN
from dataset import Poemsets


def generate(seeds, model, sentence_len=7, method=2, topN=3):
    """
    Generate one word one time
    :param seeds: int or list of int, the integer is the index of a word in the vocabulary
    :param model: Char RNN model object
    :param sentence_len: int, total length of the sentence
    :param method: 1 for generating according to all previous words; 2 for generating according to only the nearest word
    :param topN: int, the nest word is chosen from the top k candidates
    :return: seq of indexes

    PS: At present, method 1 does not have a readable result for the combinations of words haven't been saw in training data.
    """
    if not isinstance(seeds, list):
        inputs = copy.deepcopy([seeds])
    else:
        inputs = copy.deepcopy(seeds)

    outputs = inputs
    hidden = model.init_hidden(1)
    for i in range(sentence_len - len(inputs)):
        r, hidden = model.forward(Variable(torch.LongTensor([inputs])), hidden)
        topv, topi = r.data.topk(topN)  # topk returns (values, indexes)
        chosen1 = np.random.choice(topi.numpy()[0])
        outputs.append(int(chosen1))
        if method == 1:
            inputs = outputs
        elif method == 2:
            inputs = [outputs[-1]]

    return outputs


def giveme_a_poetry(seeds, model, sonnet='J', type=7, num_lines=None):
    """

    :param sonnet: str, 'J' for jueju a poem form of 4 lines , 'L' for lvshi a poem form of 8 lines
    :param type: int, 5-characters or 7-characters
    :param num_lines: int, if neither jueju nor lvshi, you can specific a length
    :return:
    """
    if num_lines:
        num_lines = num_lines
    else:
        num_lines = 8 if sonnet == "L" else 4

    lines = []
    hidden = model.init_hidden(batch_size=1)
    for nl in range(num_lines):
        l = generate(seeds=seeds, model=model, sentence_len=type)
        lines.append(l)
        r, hidden = model(Variable(torch.LongTensor([[l[-1]]])), hidden)
        topv, topi = r.data.topk(1)  # topk returns (values, indexes)
        seeds = topi[0][0]

    return lines


def giveme_an_acrostic_poetry(seeds, model, sentence_len=7):
    lines = []
    for seed in seeds:
        l = generate(seed, model, sentence_len=sentence_len, method=2)
        lines.append(l)
    return lines


if __name__ == '__main__':
    # =======================#
    # prepare data and model #
    # =======================#
    mode = "7"
    poemset = Poemsets(os.path.join(config.dir_poemlm, "qts_{m}.txt".format(m=mode)))
    char_rnn = CharRNN(len(poemset.vocab), embedding_dim=256, hidden_size=256)
    checkpoint = torch.load(os.path.join(config.dir_chkpt, mode, "59.chkpt"))
    char_rnn.load_state_dict(checkpoint)

    poem = giveme_a_poetry(poemset.word2idx["春"], char_rnn, sonnet="J")
    print([[poemset.idx2word[c] for c in l] for l in poem])

    seeds = [poemset.word2idx[s] for s in "爱我中华"]
    acrostic_poetry = giveme_an_acrostic_poetry(seeds, char_rnn, sentence_len=7)
    print([[poemset.idx2word[c] for c in l] for l in acrostic_poetry])

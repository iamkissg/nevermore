# -*- coding: utf-8 -*-

import copy
import os
import random

import numpy as np
import torch
from scipy.spatial.distance import cosine
from torch.autograd import Variable

import config
from dataset import Poemsets
from firstline import make_firstline
from model import PoetryGenerator
import util

vocab = util.load_vocab()
word2idx, idx2word = util.load_word2idx_idx2word(vocab)
pingshuiyun = util.read_pingshuiyun(use_index=True)
word2vec = util.load_word2vec()


def giveme_a_poetry(firstline, generator, word2vec, pattern, n_candicates=25):
    """

    :param firstline:
    :param generator:
    :param word2vec:
    :param pattern:
    :return:
    """

    lines = [firstline]  # lines represent the poem
    for l in range(len(pattern) - len(lines)):
        # ========================================
        # choose 1st word for each following line
        # ========================================

        # to be honest, I have no idea how to choose the 1st word,
        # so just find the most similar word to previous one
        word = idx2word[lines[-1][0]]  # for word2vec use str, here we have to convert index to word
        most_similar_words = [word2idx[p[0][0]] for p in word2vec.wv.most_similar(positive=word, topn=100)]
        most_similar_words = [  # remove those dissatisfy Ping/Ze
            w for w in most_similar_words
            if w in pingshuiyun.keys() and (pattern[l + 1][0] == 0 or (pattern[l + 1][0] == -1) ^ pingshuiyun[w])
        ]
        current_line = [random.sample(most_similar_words[:10], 1)[0]]  #

        # TODO: Antithesis (前后对仗)
        # temporarily cancel the feature by setting probability < 0.0
        if len(lines) in (1, 3) and random.random() < 0.0:
            pass
            # previous_line = [idx2word[w] for w in lines[l]]
            # for iw, w in enumerate(previous_line[1:], start=1):
            #     word_candicates = word2vec.wv.most_similar(
            #         positive=[previous_line[0], w],
            #         negative=[idx2word[current_line[0]]], topn=200, restrict_vocab=2000)
            #     word_candicates = [wc[0] for wc in word_candicates]
            #     word_candicates = [
            #         w for w in word_candicates
            #         if word2idx[w] in pingshuiyun.keys() and (pattern[l + 1][iw] == 0 or pattern[l + 1][iw] == -1) ^
            #            pingshuiyun[word2idx[w]]
            #     ]
            #     current_line.append(word2idx[word_candicates[0]])
        else:
            candicates = []
            # generate candicate lines
            for n in range(n_candicates):
                L = copy.deepcopy(current_line)
                hidden = generator.init_hidden(batch_size=1)
                for w in range(len(pattern[0]) - 1):
                    generated, hidden = generator.forward(w=Variable(torch.LongTensor([L[-1]])),
                                                          s=Variable(torch.LongTensor(lines)),
                                                          hidden=hidden)
                    topv, topi = generated.data.topk(100)
                    topi = [int(i) for i in topi.numpy()[0] if  # remove those dissatisfy Ping/Ze
                            i in pingshuiyun.keys() and (
                                pattern[l + 1][w + 1] == 0 or (pattern[l + 1][w + 1] == -1) ^ pingshuiyun[i])]
                    try:
                        L.append(random.sample(topi[:5], 1)[0])
                    except:
                        break
                candicates.append(L)

            word_candicates = [[idx2word[idx] for idx in candicate] for candicate in candicates]

            # evaluate inline (句内) cohesion scores, just the same in firstline
            cohesion_scores = {}
            for wcandicate in word_candicates:
                if len(wcandicate) == 7:
                    cohesion_scores["".join(wcandicate)] = sum([
                        word2vec.wv.n_similarity(wcandicate[:2], wcandicate[2:4]),
                        word2vec.wv.n_similarity(wcandicate[2:4], wcandicate[4:])])
                elif len(wcandicate) == 5:
                    cohesion_scores["".join(wcandicate)] = word2vec.wv.n_similarity(wcandicate[:2], wcandicate[2:])

            cohesion_scores = sorted(cohesion_scores.items(), key=lambda t: t[1], reverse=True)

            # TODO: more candicate lines, and evaluate interline cohesion
            current_line = [word2idx[w] for w in cohesion_scores[0][0]]  # choose the best one
            lines.append(current_line)
    return lines


if __name__ == '__main__':
    # =======================#
    # prepare data and model #
    # =======================#
    poemset = Poemsets(os.path.join(config.dir_rnnpg, "partitions_in_Table_2", "rnnpg", "qtrain_7"))
    pg = PoetryGenerator(vocab_size=len(vocab), embedding_dim=256, rnn_hidden_size=256, rnn_num_layers=3,
                         tie_weights=True)

    checkpoint = torch.load(os.path.join(config.dir_chkpt, "7", "70_new2.chkpt"))
    pg.load_state_dict(checkpoint)

    firstline, _, pattern, mutual_info = make_firstline("7 诗")  # first line, its mode, pattern, mutual information
    ifirstline = [word2idx[w] for w in firstline]  # represent as list of indexes

    poem = giveme_a_poetry(ifirstline, pg, word2vec, pattern=pattern)
    print(poem)
    print([[idx2word[c] for c in l] for l in poem])

# -*- coding: utf-8 -*-

import copy
import os
import random
from queue import Queue

import torch
from torch.autograd import Variable
import kenlm

import config
from dataset import Poemsets
from model import PoetryGenerator
import util

vocab = util.load_vocab()
commons = vocab[:2500]  # most frequent words
word2idx, idx2word = util.load_word2idx_idx2word(vocab)
icommons = [word2idx[w] for w in commons]
pingzes, yuns = util.read_pingshuiyun(use_index=True)
word2vec = util.load_word2vec()
LM = os.path.join(os.path.dirname(__file__), '..', 'data', 'rnnpg_data_emnlp-2014',
                  'partitions_in_Table_2', 'poemlm', 'qts.klm')
kmodel = kenlm.Model(LM)

poemset = Poemsets(os.path.join(config.dir_rnnpg, "partitions_in_Table_2", "rnnpg", "qtrain_7"))
pg = PoetryGenerator(vocab_size=len(vocab), embedding_dim=256, rnn_hidden_size=256, rnn_num_layers=3,
                     tie_weights=True)
checkpoint = torch.load(os.path.join(config.dir_chkpt, "7", "383_new2.chkpt"))
pg.load_state_dict(checkpoint)


def get_yun(w):
    for head, words in yuns.items():
        if w in words:
            return head
    raise ValueError("Can not find Yun for {w}".format(w=w))


def giveme_a_poetry(firstline, pattern, generator=pg, firstwords=None, word2vec=word2vec, n_candicates=2):
    """

    :param firstline:
    :param generator:
    :param word2vec:
    :param pattern:
    :return:
    """

    lines = [firstline]  # lines represent the poem
    yun_head = get_yun(firstline[-1])
    generated_firstwords = []

    for l in range(len(pattern) - len(lines)):
        # ========================================
        # choose 1st word for each following line
        # ========================================

        if firstwords:
            current_line = [firstwords[l]]
        else:
            # to be honest, I have no idea how to choose the 1st word,
            # so just find the most similar word to previous one
            word = idx2word[lines[-1][0]]  # for word2vec use str, here we have to convert the first word in last line from index to word
            generated_firstwords.append(word)
            most_similar_words = [word2idx[p[0][0]] for p in word2vec.wv.most_similar(positive=word, topn=250)]
            most_similar_words = [  # remove those dissatisfy Ping/Ze.
                w for w in most_similar_words
                if w in icommons and w in pingzes.keys() and (pattern[l + 1][0] == 0 or (pattern[l + 1][0] == -1) ^ pingzes[w])
            ]
            # current_line = [random.sample(most_similar_words[:5], 1)[0]]  #
            ken_scores = [kmodel.score(" ".join(generated_firstwords) + " " + w) for w in (idx2word[i] for i in most_similar_words)]
            rd_ken_scores = random.sample(ken_scores, 3)# random drop
            chosen = ken_scores.index(max(ken_scores))
            current_line = [chosen]

        # TODO: Antithesis (前后对仗)
        # temporarily cancel the feature by setting probability < 0.0
        if len(lines) in (1, 3) and (len(set(lines[-1])) < len(lines[-1])) and random.random() < 0.0:
            previous_line = [idx2word[w] for w in lines[l]]
            for iw, w in enumerate(previous_line[1:], start=1):  # start from the second word
                word_candicates = word2vec.wv.most_similar(
                    positive=[previous_line[0], w],
                    negative=[idx2word[current_line[0]]], topn=500, restrict_vocab=2000)
                word_candicates = [wc[0] for wc in word_candicates]
                word_candicates = [
                    w for w in word_candicates
                    if word2idx[w] in pingzes.keys() and (pattern[l + 1][iw] == 0 or pattern[l + 1][iw] == -1) ^ pingzes[word2idx[w]]
                ]
                current_line.append(word2idx[word_candicates[0]])
            lines.append(current_line)
        else:
            candicates = Queue()
            candicate_scores = Queue()
            hiddens = Queue()

            # ===========
            # beam search
            # ===========

            # init candicates queue and hidden states queue
            candicates.put(copy.deepcopy(current_line))
            candicate_scores.put([1])
            hiddens.put(generator.init_hidden(batch_size=1))
            # import pdb
            # pdb.set_trace()
            for e in range(len(pattern[0]) - 1):
                for n in range(n_candicates ** e):
                    cdct = candicates.get()
                    score = candicate_scores.get()
                    hidden = hiddens.get()
                    generated, hidden = generator.forward(w=Variable(torch.LongTensor([cdct[-1]])),
                                                          s=Variable(torch.LongTensor(lines)),
                                                          hidden=hidden)
                    if e == len(pattern[0]) - 2 and len(lines) in (1, 3):
                        topv, topi = generated.data.topk(3500)
                        topi = [int(i) for i in topi.numpy()[0] if  # remove those dissatisfy Ping/Ze
                                i.item() in yuns[yun_head] and i in pingzes.keys() and (
                                pattern[l + 1][len(cdct)] == 0 or (pattern[l + 1][len(cdct)] == -1) ^ pingzes[i])]
                    else:
                        topv, topi = generated.data.topk(2500)
                        topi = [int(i) for i in topi.numpy()[0] if  # remove those dissatisfy Ping/Ze
                                i in pingzes.keys() and (
                                pattern[l + 1][len(cdct)] == 0 or (pattern[l + 1][len(cdct)] == -1) ^ pingzes[i])]

                    try:
                        for m in range(n_candicates):
                            lth_index = topi[m]
                            copied_cdct = copy.deepcopy(cdct)
                            copied_cdct.append(lth_index)
                            copied_score = copy.deepcopy(score)
                            copied_score.append(generated.data.numpy()[0][lth_index])
                            candicates.put(copied_cdct)
                            candicate_scores.put(copied_score)
                            hiddens.put(hidden)
                    except Exception as ecpt:
                        raise ecpt
            candicates = [candicates.get() for q in range(candicates.qsize())]
            candicate_scores = [candicate_scores.get() for q in range(candicate_scores.qsize())]

            word_candicates = [[idx2word[idx] for idx in candicate] for candicate in candicates]

            candicate_scores = [sum(cs) for cs in candicate_scores]
            min_score = min(candicate_scores)
            score_range = max(candicate_scores) - min(candicate_scores)
            normalized_scores = [(cs - min_score) / score_range for cs in candicate_scores]

            cohesion_scores = [kmodel.score(" ".join(words)) for words in word_candicates]
            cohesion_min_score = min(cohesion_scores)
            cohesion_score_range = max(cohesion_scores) - cohesion_min_score
            normalized_cohesion_scores = [(cs - cohesion_min_score) / cohesion_score_range for cs in cohesion_scores]

            scores = [a * 0.15 + b * 0.85 for a, b in zip(normalized_scores, normalized_cohesion_scores)]
            max_score = max(scores)
            max_index = scores.index(max_score)
            chosen = candicates[max_index]

            # TODO: more candicate lines, and evaluate interline cohesion
            # current_line = [word2idx[w] for w in cohesion_scores[0][0]]  # choose the best one
            current_line = chosen
            lines.append(current_line)
    return lines


if __name__ == '__main__':
    import argparse
    from firstline import make_firstline

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', '--qtype', type=int, default=7,
                           help="Specify 5-characters quatrain or 7-characters quatrain.")
    argparser.add_argument('-w', '--qtopic', type=str, default="春",
                           help="Specify topic for quatrain.")
    args = argparser.parse_args()

    fline, _, pattern, mutual_info = make_firstline(str(args.qtype) + " " + args.qtopic)  # first line, its mode, pattern, mutual information
    ifline = [word2idx[w] for w in fline]  # represent as list of indexes
    poem = giveme_a_poetry(ifline, pattern=pattern)
    print("\r\n".join(["".join([idx2word[c] for c in l]) for l in poem]))


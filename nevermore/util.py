# -*- coding: utf-8 -*-
import os

from gensim.models import Word2Vec
import numpy as np

import config


def load_vocab():
    with open(config.path_vocab) as f:
        vocab = f.read().split("\n")
    return vocab


def load_word2idx_idx2word(vocab=None):
    if not vocab:
        vocab = load_vocab()
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    return word2idx, idx2word


def word2idx_or_selfmap(index=False):
    """a helper function"""
    if index:
        word2idx, _ = load_word2idx_idx2word()
        return lambda w: word2idx[w]
    else:
        return lambda w: w  # self-mapping, for consistency


def load_word2vec():
    """load gensim word2vec model"""
    return Word2Vec.load(os.path.join(config.dir_data, "word2vec.model"))


def load_embedding(filename, use_index=False):
    """load embedding as a dict"""

    # load embedding into memory, skip first line
    with open(filename, "r") as f:
        lines = f.readlines()[1:]

    F = word2idx_or_selfmap(use_index)

    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        # for example, "不": [0.650151 -0.063599 ... -1.016167 0.373945]
        embedding[F(parts[0])] = np.asarray(parts[1:], dtype='float32')
    return embedding


def get_weight_matrix(embedding, word2idx, embedding_dim=256):
    """create a weight matrix for the Embedding layer from a loaded embedding"""

    vocab_size = len(word2idx)
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, embedding_dim))  # vocab_size x embedding_dim

    # step vocab, store vectors
    for word, i in word2idx.items():
        weight_matrix[i] = embedding.get(word)  # embedding is word-vector dict
    return weight_matrix


def read_pingshuiyun(use_index=False):
    """get the Ping/Ze dict of each word based on '平水韵 (Ping Shui Yun)' """
    pingzes = {}  # dict of Pingze
    yuns = {}  # dict of Yun

    is_ping = False
    word2idx, _ = load_word2idx_idx2word()
    included_words = word2idx.keys()

    F = word2idx_or_selfmap(use_index)

    with open(config.path_pingshuiyun, "r") as f:
        # the result of f.readlines() has end "\n"
        for line in f.read().split("\n")[:-1]:  # the last line is empty
            if line[0] == '/':
                is_ping = not is_ping
                continue
            for word in line:
                # there may be some word in 'Ping Shui Yun' but not in vocab
                # one problem is some words have a variety of pronunciation, like "不"
                # these words will be labeled as Ze.
                if word in included_words:
                    pingzes[F(word)] = is_ping
                else:
                    continue
            yuns[F(line[0])] = [F(w) for w in list(line) if w in included_words]
    return pingzes, yuns


def read_shixuehanying():
    """get topic-class mapping as well as class-phrase mapping according to '诗学含英 (Shi Xue Han Ying)'"""
    topic_class = {}
    class_phrase = {}

    current_topic = ""
    with open(config.path_shixuehanying, "r") as f:
        for line in f.read().split("\n")[:-1]:
            if line[0] == "<":  # <begin> or <end>
                if line[1] == "b":  # <begin> case, skip <end> case
                    current_topic = line.split("\t")[2]
                    topic_class[current_topic] = []
            else:
                items = line.split("\t")
                klass = items[1]
                topic_class[current_topic].append(klass)  # topic-class mapping
                phrases = items[2].split(" ")
                class_phrase[items[1]] = phrases

    return topic_class, class_phrase


if __name__ == '__main__':
    # ===== #
    # test  #
    # ===== #
    print(load_embedding(config.path_embedding))

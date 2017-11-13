# -*- coding: utf-8 -*-

from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Poemsets(Dataset):
    def __init__(self, fn):
        with open(fn) as f:
            poems = f.readlines()
        word_freqs = Counter("".join(poems).replace("\n", ""))
        word_freqs = sorted(word_freqs.items(), key=lambda x: -x[1])
        self.vocab, _ = zip(*word_freqs)
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}  # word to index
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}  # word to index
        self.data = torch.LongTensor([[self.word2idx[w] for w in p.strip()] for p in poems])

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    import os
    sevens = os.path.join(os.path.dirname(__file__), os.pardir, "data", "rnnpg_data_emnlp-2014", "partitions_in_Table_2", "poemlm", "qts_7.txt")
    poems7set = Poemsets(sevens)
    print(poems7set.vocab)
    print(poems7set.word2idx)
    print(poems7set.idx2word)
    print(poems7set.data)

    poems7loader = DataLoader(poems7set, batch_size=1024, shuffle=True)
    for i, p in enumerate(poems7loader):
        print(p)
        if i == 1:
            break

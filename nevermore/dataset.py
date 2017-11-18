# -*- coding: utf-8 -*-

import os

import torch
from torch.utils.data import Dataset, DataLoader

import config
from util import load_word2idx_idx2word, load_vocab


class Poemsets(Dataset):
    def __init__(self, fn):
        with open(fn) as f:
            poems = f.readlines()

        word2idx, _ = load_word2idx_idx2word()
        self.data = torch.LongTensor([
            [[word2idx[w] for w in line.split(" ")] for line in poem.strip("\n").split("\t")]
            for poem in poems if "<R>" not in poem
        ])

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    import os

    sevens = os.path.join(os.path.dirname(__file__), os.pardir, "data", "rnnpg_data_emnlp-2014",
                          "partitions_in_Table_2", "rnnpg", "qtrain_7")
    poems7set = Poemsets(sevens)
    print(poems7set.data)

    poems7loader = DataLoader(poems7set, batch_size=1024, shuffle=True)
    for i, p in enumerate(poems7loader):
        print(p)
        if i == 1:
            break

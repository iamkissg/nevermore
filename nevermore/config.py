# -*- coding: utf-8 -*-

import os

# dir, for convenience
dir_chkpt = os.path.join("checkpoints")
dir_data = os.path.join(os.path.dirname(__file__), os.pardir, "data")
dir_rnnpg = os.path.join(dir_data, "rnnpg_data_emnlp-2014")
dir_poemlm = os.path.join(dir_rnnpg, "partitions_in_Table_2", "poemlm")

# file path
path_pingshuiyun = os.path.join(dir_data, "pingshuiyun.txt")  # Pingshuiyun contains the Ping/Ze of words, incomplete
path_shixuehanying = os.path.join(dir_data, "shixuehanying.txt")  # Shixuehanying contains the category of words
path_embedding = os.path.join(dir_data, "embedding_word2vec.txt")
path_vocab = os.path.join(dir_data, "vocab.txt")

# The most common tonals for quatrain
# (0 for either, 1 for Ping, -1 for Ze)
QUATRAIN_5 = [
    [[0, -1, 1, 1, -1], [1, 1, -1, -1, 1], [0, 1, 1, -1, -1], [0, -1, -1, 1, 1]],  # 首句仄起仄收
    [[0, -1, -1, 1, 1], [1, 1, -1, -1, 1], [0, 1, 1, -1, -1], [0, -1, -1, 1, 1]],  # 首句仄起平收
    [[0, 1, 1, -1, -1], [0, -1, -1, 1, 1], [0, -1, 1, 1, -1], [1, 1, -1, -1, 1]],  # 首句平起仄收
    [[1, 1, -1, -1, 1], [0, -1, -1, 1, 1], [0, -1, 1, 1, -1], [1, 1, -1, -1, 1]]  # 首句平起平收
]

QUATRAIN_7 = [
    [[0, 1, 0, -1, -1, 1, 1], [0, -1, 1, 1, -1, -1, 1], [0, -1, 0, 1, 1, -1, -1], [0, 1, 0, -1, -1, 1, 1]],  # 首句平起平收
    [[0, 1, 0, -1, 1, 1, -1], [0, -1, 1, 1, -1, -1, 1], [0, -1, 0, 1, 1, -1, -1], [0, 1, 0, -1, -1, 1, 1]],  # 首句平起仄收
    [[0, -1, 1, 1, -1, -1, 1], [0, 1, 0, -1, -1, 1, 1], [0, 1, 0, -1, 1, 1, -1], [0, -1, 1, 1, -1, -1, 1]],  # 首句仄起平收
    [[0, -1, 0, 1, 1, -1, -1], [0, 0, -1, -1, -1, 1, 1], [0, 1, 0, -1, 1, 1, -1], [0, -1, 1, 1, -1, -1, 1]]  # 首句仄起仄收
]

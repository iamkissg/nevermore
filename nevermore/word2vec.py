# -*- coding: utf-8 -*-

import os

from gensim.models import Word2Vec

import config


def load_file(filename):
    with open(filename, "r") as f:
        text = f.read()
    return text


# ==============
# Prepare corpus
# ==============
# poems object is a str
poems = load_file(os.path.join(config.dir_rnnpg, "partitions_in_Table_2", "rnnpg", "qtotal"))

# str -> list,
# one sentence in one poem is one line
sentences = [line.replace(" ", "") for p in poems.split("\n") if "<R>" not in p for line in p.strip("\n").split("\t")]

# for follow-up convenience, try to convert word to index representation,
# but get confused for it's possible to discriminate [1011] is [101, 1] or [10, 11]

# =====================
# Train word2vec model
# =====================
model = Word2Vec(sentences, size=256, window=3, workers=16, min_count=1, iter=10)

# Save model, vocab and embedding vectors
model.save(os.path.join(config.dir_data, "word2vec.model"))
model.wv.save_word2vec_format(os.path.join(config.dir_data, "embedding_word2vec.txt"), binary=False)
with open(os.path.join(config.dir_data, "vocab.txt"), "w") as f:
    f.write("\n".join(model.wv.index2word))

# to see the effect
print(model.wv.similarity("春", "秋"))
print(model.wv.similarity("山", "水"))
print(model.wv.similarity("白", "帝"))
print(model.wv.most_similar(positive=["春", "花"], negative=["秋"]))
print(model.wv.most_similar(positive=["花"]))

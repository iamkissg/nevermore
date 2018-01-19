# -*- coding: utf-8 -*-

import logging.handlers
import os
import random
import re

import requests
import kenlm

import config
from util import load_embedding, word2idx_or_selfmap, read_pingshuiyun, read_shixuehanying, load_word2vec

# ============
# log setting

# Because 'Ping Shui Yun' is incomplete, set log for logging missing word
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.handlers.TimedRotatingFileHandler("{0}/{1}".format(config.dir_data, "Logging"), when="D")
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)
# ************

RE_RHYMENAME = re.compile(r"""rhymeName">(.+?)<""")


# because words in 'Shi Xue Han Ying' are represented as str, we read_pingshuiyun(use_index=False)
# it's more convenient to use str.
pingzes, yuns = read_pingshuiyun(use_index=False)
topic_class, class_phrase = read_shixuehanying()
class_topic = {c: t for t, cs in topic_class.items() for c in cs}
LM = os.path.join(os.path.dirname(__file__), '..', 'data', 'rnnpg_data_emnlp-2014',
                  'partitions_in_Table_2', 'poemlm', 'qts.klm')
kmodel = kenlm.Model(LM)


def generate_one_candicate(phrases, classes, length=7, pattern=-1):
    """
    :param classes:
    :param length:
    :param pattern:
    :return:
    """
    candicate = []  # list of words
    pos = len(candicate)  # position of last word in candicate

    # determine the PATTERN for the poetry
    if pattern == -1:
        pattern = random.randint(0, 3)
    PATTERN = config.QUATRAIN_7[pattern] if length == 7 else config.QUATRAIN_5[pattern]

    # the concept is learned from InfoGAN
    # the mutual information between first line and topic (TODO: use mutual information in the whole generation)
    mutual_info = 1.0

    # record length of used phrases
    mode = [0]

    used_phrases = set()  # record used phrases
    if length == 7:
        # allowed phrases for 7-character poem, 2-2-3, 2-5 or 4-3
        avoid_phrases = {p for p in phrases if len(p) != 2 and len(p) != 4}
    else:
        # allowed phrases for 5-character poem, 2-3 or 5
        avoid_phrases = {p for p in phrases if len(p) != 2 and len(p) != 5}

    phrase = ""
    while len(candicate) < length:
        try:
            phrase = random.sample(phrases.difference(avoid_phrases), 1)[0]  # random.sample returns a list, so [0]
        except ValueError as e:
            # TODO, extend to better class. For example "荷" -> [莲花", "白莲花", "荷花"]
            # the phrases under current classes is not enougth, get more phrases under current topics
            if e.args[0] == "Sample larger than population":
                # search other classes under their topics
                topics = [class_topic[c] for c in classes]
                topic = random.sample(topics, 1)[0]  # choose 1 of the father topics
                logging.info("USE father topic {t}".format(t=topic))
                classes = topic_class[topic]  # do not overwrite the classes

                for item in classes:
                    phrases = phrases.union(set(class_phrase.get(item)))

                # reconstruct avoid_phrases
                if length - len(candicate) == 3:
                    avoid_phrases = {p for p in phrases if len(p) != 3}.union(used_phrases)
                    mutual_info -= 0.3
                elif length - len(candicate) == 5:
                    avoid_phrases = {p for p in phrases if len(p) != 2 and len(p) != 5}.union(used_phrases)
                    mutual_info -= 0.5
                elif length - len(candicate) == 7:
                    # the condition len(candicate) == 0 does not equal to this one
                    # maybe at the very beginning, there is no satisfied phrases
                    avoid_phrases = {p for p in phrases if len(p) != 2 and len(p) != 4}
                    mutual_info -= 0.8
        else:
            for i, word in enumerate(phrase):
                try:
                    # check ping/ze
                    if (PATTERN[0][pos + i] == 0) or ((PATTERN[0][pos + i] == -1) ^ pingzes[word]):
                        continue
                    else:
                        avoid_phrases.add(phrase)
                        break
                except KeyError:
                    # this error results from the incompleteness of pingshuiyun.txt, find the ping/ze on web
                    # resp = requests.get("http://gd.sou-yun.com/QR.aspx?c={word}".format(word=word))
                    # rhymename = RE_RHYMENAME.findall(resp.text)[0][-1]
                    # logging.info("{word} -> {rhymename}.".format(word=word, rhymename=rhymename))
                    continue
            else:
                candicate.extend(list(phrase))
                mode.append(pos + len(phrase))
                pos = len(candicate)  # move pos to the end of the current words
                used_phrases.add(phrase)
                if length - len(candicate) == 5:
                    avoid_phrases = set([p for p in phrases if len(p) != 2 and len(p) != 5]).union(used_phrases)
                else:
                    avoid_phrases = set([p for p in phrases if len(p) != 3]).union(used_phrases)
    return candicate, mode, PATTERN, mutual_info


def select_best(cmms, k=1):
    """cmms - (candicate, mode, mutual info) pairs"""

    cohesion_scores = {}
    mutual_info_scores = {}
    for candicate, mode, mutual_info in cmms:
        try:
            cohesion_scores["".join(candicate)] = kmodel.score(" ".join(candicate))
        except KeyError:  # some rarely-used Chinese characters maybe not included in word2vec model, just let it go.
            cohesion_scores["".join(candicate)] = -100
        mutual_info_scores["".join(candicate)] = mutual_info

    scores = {k: cohesion_scores[k] + mutual_info_scores[k] for k in cohesion_scores.keys()}

    topk = sorted(scores.items(), key=lambda t: t[1], reverse=True)[:k]  # descending
    chosen = random.sample(topk, 1)[0]

    return list(chosen[0])


def make_firstline(inputs, n_candicates=66):
    """make first line for a poem"""

    # inputs can be a whitespace delimited str whose first char should be an integer, or a sequence
    if isinstance(inputs, str):
        inputs = inputs.split()

    # length determines the poem pattern,
    # in general, it is 5 or 7 for 5-character or 7-character respectively
    length = int(inputs[0])

    # phrases for given classes
    phrases = set()
    for item in inputs[1:]:
        # TODO: Match arbitrary input items to most like class in 'Shi Xue Han Ying'
        if item in class_topic.keys():
            phrases = phrases.union(set(class_phrase.get(item)))

    cpms = []  # candicate line (first sentence), its pattern, its mutual information
    while len(cpms) < n_candicates:
        candicate, mode, pattern, mutual_info = generate_one_candicate(phrases, classes=inputs[1:], length=length)
        cpms.append((candicate, mode, pattern, mutual_info))

    # when select best candicate, the pattern is not needed.
    best = select_best([(cpm[0], cpm[1], cpm[3]) for cpm in cpms])
    index = [cpm[0] for cpm in cpms].index(best)

    return cpms[index]


if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', '--qtype', type=int, default=7,
                           help="Specify 5-characters quatrain or 7-characters quatrain.")
    argparser.add_argument('-w', '--qtopic', type=str, default="春",
                           help="Specify topic for quatrain.")
    args = argparser.parse_args()

    print("".join(make_firstline(str(args.qtype) + " " + args.qtopic)[0]))

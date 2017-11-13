# -*- coding: utf-8 -*-

import os
import glob
import re

# TODO: For other data sources
#
# Zhengzhou University's data
# punctuation = re.compile(r"""[，。]""")
# dir_zzu_qts = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "quantangshi_zhengzhou")
# zzu_qts = glob.iglob(os.path.join(dir_zzu_qts, "js_*_ns_*.txt"))

dir_rnnpg = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rnnpg_data_emnlp-2014")


def prepare_data():
    """
    This function is to get 7-characters poem and 5-characters poem for convenience.
    """
    with open(os.path.join(dir_rnnpg, "partitions_in_Table_2", "poemlm", "qts_tab.txt")) as f:
        lines = f.readlines()
        lines = sorted(lines, key=lambda l: len(l))
        lines = [l.replace(" ", "") for l in lines]
        c5 = [l for l in lines if len(l) == 6]
        c7 = [l for l in lines if len(l) == 8]  # severn-characters poem
        with open(os.path.join(dir_rnnpg, "partitions_in_Table_2", "poemlm", "qts_5.txt"), "w") as wf:
            wf.writelines(c5)
        with open(os.path.join(dir_rnnpg, "partitions_in_Table_2", "poemlm", "qts_7.txt"), "w") as wf:
            wf.writelines(c7)


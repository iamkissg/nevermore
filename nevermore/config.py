# -*- coding: utf-8 -*-

import os


# dir, for convenient
dir_chkpt = os.path.join("checkpoints")
dir_data = os.path.join(os.path.dirname(__file__), os.pardir, "data")
dir_rnnpg = os.path.join(dir_data, "rnnpg_data_emnlp-2014")
dir_poemlm = os.path.join(dir_rnnpg, "partitions_in_Table_2", "poemlm")

# AI 诗人 Nevermore

> 本项目在 Ubuntu 16.04 测试通过, 原则上在 Linux/Mac 上也能运行, 但不保证在 Windows 下的使用.
> 代码于 2017/10 发布于 GitHub, 注释基本为英文注释.

## 项目结构

项目的结构如下所示:

```txt
.
├── data
│   ├── embedding_word2vec.txt
│   ├── pingshuiyun.txt
│   ├── pingze.txt
│   ├── rnnpg_data_emnlp-2014
│   │   └── partitions_in_Table_2
│   │       ├── poemlm
│   │       │   ├── qts_5.txt
│   │       │   ├── qts_7.txt
│   │       │   ├── qts.arpa
│   │       │   └── qts.klm
│   │       └── rnnpg
│   │           ├── qtest
│   │           ├── qtest_5
│   │           ├── qtest_7
│   │           ├── qtotal
│   │           ├── qtrain
│   │           ├── qtrain_5
│   │           ├── qtrain_7
│   │           ├── qvalid
│   │           ├── qvalid_5
│   │           └── qvalid_7
│   ├── shixuehanying.txt
│   ├── vocab.txt
│   └── word2vec.model
├── nevermore
│   ├── checkpoints
│   │   └── 7
│   │       └── 383_new2.chkpt
│   ├── config.py
│   ├── dataset.py
│   ├── firstline.py
│   ├── generate.py
│   ├── __init__.py
│   ├── model.py
│   ├── train.py
│   ├── util.py
│   └── word2vec.py
└── README.md
```

* `data`为数据目录, 包含了除 Model Checkpoints 之外所有本项目使用到的数据;
    * `pingshuiyun.txt` - 平水韵
    * `shixuehanying.txt` - 诗学含英
    * `pingze.txt` - 8 种常用格律
    * `embedding_word2vec.txt` - 使用 Gensim 得到的 Word embedding 文件
    * `vocab.txt` - 使用 Gensim 得到的词汇表文件
    * `word2vec.model` - 使用 Gensim 得到的 Word2Vec 模型
    * `rnnpg_data_emnlp-2014` - 本项目使用了 [Chinese Poetry Generation with Recurrent Neural Networks](http://aclweb.org/anthology/D/D14/D14-1074.pdf) 提供的诗词数据, 并未修改原目录结构
* `nevermore` 下为本项目的源代码, 以及 Model Checkpoints
    * `model.py` - 模型定义脚本
    * `train.py` - 模型训练脚本
    * `generate.py` - 模型脚本, 使用训练好的模型作诗
    * `firstline.py` - 生成诗的首句
    * `dataset.py` - 创建数据集的脚本
    * `config.py` - 保存了本项目的一些基本配置项
    * `util.py` - 常用函数的集合
    * `word2vec.py` - 用于生成 word2vec.model, embedding_word2vec.txt, vocab.txt
    
## 依赖

* Python (3.5)
* Pytorch (0.2.0) (官方未提供 Windows 库)
* NumPy (1.13.3)
* KenLM
* Gensim (3.2.0)

## 安装

KenLM 的安装比较复杂, 请参考[这篇文章](http://thegrandjanitor.com/2015/12/28/using-arpa-lm-with-python/)

其他 Python 包安装非常简单, 以 pip 为例:

安装 Pytorch (CUDA 8.0)

```shell
pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl
```

安装 Gensim

```shell
pip3 uninstall numpy
pip3 install gensim==3.2.0
```

Gensim 和 Pytorch 对 NumPy 都有依赖, 因此执行完上面两条命令, 已经安装好 NumPy. 若发现 NumPy 版本不同影响到程序运行, 可执行运行命令:

```shell
pip3 uninstall numpy
pip3 install numpy==1.13.3
```

## 使用

执行命令`python nevermore/firstline.py -n 5 -w 春`将生成首句诗. `-n/--qtype`用于指定每句字数, 一般为五言或七言; `-w/--qtopic`指定用于作诗的意象, 可使用的意象详见[data/shixuehanying.txt](data/shixuehanying.txt).

在`nevermore/checkpoints/7/`目录下, 已经提供了一个 Model Checkpoint `383_new2.chkpt`, 可直接使用.

执行命令`python nevermore/generate.py -n 7 -w 诗`将生成一首完整的, 以**诗**为意象的七言绝句.

---

若希望从头开始训练模型, 可依照以下步骤:

1. 执行`python nevermore/word2vec.py`, 得到 word2vec.model, embedding_word2vec.txt, vocab.txt
2. 执行`python nevermore/train.py`, 训练模型, 不失一般性, 使用 CPU 而非 GPU, 因此训练较慢.
3. 执行`python nevermore/generate.py -n 7 -w 诗`, 作诗


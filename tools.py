#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: tools.py
@time:2022/04/25
@description:
"""
import json


def trans():
    """
    将chinese-snli中的数据转成可阅读形式
    :return:
    """
    file_path = ["corpus/chinese-snli/cnsd_snli_v1.0.dev.jsonl",
                 "corpus/chinese-snli/cnsd_snli_v1.0.train.jsonl",
                 "corpus/chinese-snli/cnsd_snli_v1.0.test.jsonl",
                 ]
    target_file_path = ["corpus/chinese-snli-c/dev.txt",
                        "corpus/chinese-snli-c/train.txt",
                        "corpus/chinese-snli-c/test.txt",
                        ]

    def trans_data(data_path, target_path):
        lines = []
        with open(data_path, 'r', encoding='utf8') as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                line = json.loads(line)
                lines.append(line)
        with open(target_path, 'w', encoding='utf8') as writer:
            writer.write("\n".join([json.dumps(data, ensure_ascii=False) for data in lines]))

    for f, t in zip(file_path, target_file_path):
        trans_data(f, t)


def trans_LCQMC():
    """
    将LCQMC格式数据转成dataloader默认处理格式的数据
    :return:
    """
    file_path = ["corpus/LCQMC/train.txt",
                 "corpus/LCQMC/dev.txt",
                 "corpus/LCQMC/test.txt",
                 ]
    target_file_path = ["corpus/LCQMC_S/train.txt",
                        "corpus/LCQMC_S/dev.txt",
                        "corpus/LCQMC_S/test.txt",
                        ]
    def trans_data(data_path, target_path):
        lines = []
        with open(data_path, 'r', encoding='utf8') as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                splits = line.split("\t")
                if len(splits) ==3:
                    sen1, sen2, label = splits
                    lines.append({"sentence1": sen1, "sentence2": sen2, "gold_label": label})
        with open(target_path, 'w', encoding='utf8') as writer:
            writer.write("\n".join([json.dumps(data, ensure_ascii=False) for data in lines]))

    for f, t in zip(file_path, target_file_path):
        trans_data(f, t)



if __name__ == '__main__':
    # trans()
    trans_LCQMC()

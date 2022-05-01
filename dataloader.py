#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: dataloader.py
@time:2022/04/17
@description:
"""

import json
import os
from typing import Optional, List, Dict
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import jieba


def jieba_cut(content: str):
    return [word for word in jieba.cut(content) if word]


def char_cut(content: str):
    return [char for char in list(content) if char]


class NLIDataSet(Dataset):

    def __init__(self, data_list, word2index, tag2index):
        self.word2index = word2index
        self.tag2index = tag2index
        self.data_list = self._num_data(data_list)

    def _num_data(self, data_list):
        num_data_list = []
        max_sen1_len = max([len(json_data["sentence1"]) for json_data in data_list])
        max_sen2_len = max([len(json_data["sentence2"]) for json_data in data_list])
        for dict_data in data_list:
            sentence1, sentence2 = dict_data["sentence1"], dict_data["sentence2"]
            sen1_len, sen2_len = len(sentence1), len(sentence2)
            # 有一个为空时跳过
            if not (sen2_len and sen1_len):
                continue
            sentence1_num = [self.word2index.get(word) for word in sentence1] + [self.word2index.get("<pad>")] * (
                        max_sen1_len - sen1_len)
            sentence2_num = [self.word2index.get(word) for word in sentence2] + [self.word2index.get("<pad>")] * (
                        max_sen2_len - sen2_len)
            num_data_list.append([sentence1_num, sen1_len, sentence2_num, sen2_len, self.tag2index.get(dict_data["gold_label"])])
        return num_data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)


class NLIDataModule(pl.LightningDataModule):

    def __init__(self, data_dir="corpus/chinese-snli-c", batch_size=3, experiment=False):
        super().__init__()
        self.data_path = data_dir
        self.batch_size = batch_size
        self.train_data_set, self.dev_data_set, self.test_data_set = None, None, None
        self.tag2idx, self.token2index = None, None
        self.setup()

    def _load_data(self, file_path) -> List[Dict]:
        data_list = []
        with open(file_path, 'r', encoding='utf8') as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                json_data: dict = json.loads(line)
                json_data["sentence1"] = char_cut(json_data["sentence1"])
                json_data["sentence2"] = char_cut(json_data["sentence2"])
                data_list.append(json_data)
        return data_list

    def setup(self, stage: Optional[str] = None) -> None:
        train_data_list = self._load_data(os.path.join(self.data_path, "train.txt"))
        dev_data_list = self._load_data(os.path.join(self.data_path, "dev.txt"))
        test_data_list = self._load_data(os.path.join(self.data_path, "test.txt"))

        self.char2idx = {"<pad>": 0, "<unk>": 1}
        self.tag2idx = {}
        for data_list in [train_data_list, dev_data_list, test_data_list]:
            for dict_data in data_list:
                for words in [dict_data["sentence1"], dict_data["sentence2"]]:
                    for word in words:
                        if word not in self.char2idx:
                            self.char2idx[word] = len(self.char2idx)
                if dict_data["gold_label"] not in self.tag2idx:
                    self.tag2idx[dict_data['gold_label']] = len(self.tag2idx)

        self.idx2char = {index: char for char, index in self.char2idx.items()}
        self.idx2tag = {index: value for value, index in self.tag2idx.items()}
        self.tag_size = len(self.tag2idx)
        self.vocab_size = len(self.char2idx)
        self.train_data_set = NLIDataSet(train_data_list, self.char2idx, self.tag2idx)
        self.dev_data_set = NLIDataSet(dev_data_list, self.char2idx, self.tag2idx)
        self.test_data_set = NLIDataSet(test_data_list, self.char2idx, self.tag2idx)

    @staticmethod
    def collate_fn(batch):
        sen1, sen1_len, sen2, sen2_len, y = [], [], [], [], []
        for simple in batch:
            sen1.append(simple[0])
            sen1_len.append(simple[1])
            sen2.append(simple[2])
            sen2_len.append(simple[3])
            y.append(simple[-1])
        sen1_t = torch.tensor(sen1, dtype=torch.long)
        sen1_len_t = torch.tensor(sen1_len, dtype=torch.long)
        sen2_t = torch.tensor(sen2, dtype=torch.long)
        sen2_len_t = torch.tensor(sen2_len, dtype=torch.long)
        y_t = torch.tensor(y, dtype=torch.long)
        return sen1_t, sen1_len_t, sen2_t, sen2_len_t, y_t

    def train_dataloader(self):
        return DataLoader(self.train_data_set, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_data_set, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dev_data_set, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def save_dict(self, data_dir):
        with open(os.path.join(data_dir, "index2tag.txt"), 'w', encoding='utf8') as writer:
            json.dump(self.idx2tag, writer, ensure_ascii=False)

        with open(os.path.join(data_dir, "token2index.txt"), 'w', encoding='utf8') as writer:
            json.dump(self.char2idx, writer, ensure_ascii=False)
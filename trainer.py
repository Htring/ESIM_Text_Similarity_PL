#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: trainer.py
@time:2022/04/19
@description:
"""
import json
import os
from argparse import ArgumentParser

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from model.ESIM_PL import ESIM_PL, ESIM
from dataloader import NLIDataModule, char_cut

pl.seed_everything(2022)


def training(param):
    nli_dm = NLIDataModule(data_dir=param.data_dir, batch_size=param.batch_size)
    checkpoint_callback = ModelCheckpoint(monitor='f1_score',
                                          filename="esim-{epoch:03d}-{val_loss:.3f}-{f1_score:.3f}",
                                          dirpath=param.save_dir,
                                          save_top_k=3,
                                          mode="max")
    os.makedirs(param.save_dir, exist_ok=True)
    nli_dm.save_dict(param.save_dir)
    param.vocab_size = len(nli_dm.char2idx)
    param.num_classes = len(nli_dm.tag2idx)
    model = ESIM_PL(param)
    if param.load_pre:
        model = model.load_from_checkpoint(param.pre_ckpt_path)
    logger = TensorBoardLogger("log_dir", name="esim")

    trainer = pl.Trainer(logger=logger, gpus=1,
                         callbacks=[checkpoint_callback],
                         max_epochs=param.epoch,
                         precision=16,
                         )
    trainer.fit(model=model, datamodule=nli_dm)
    nli_dm.save_dict(param.save_dir)
    trainer.test(model, nli_dm)


def model_use(param):
    def _load_dict(dir_name):
        with open(os.path.join(dir_name, 'token2index.txt'), 'r', encoding='utf8') as reader:
            token2index = json.load(reader)

        with open(os.path.join(dir_name, 'index2tag.txt'), 'r', encoding='utf8') as reader:
            index2tag = json.load(reader)

        return token2index, index2tag

    def _number_data(content):
        number_data = []
        for char in char_cut(content):
            number_data.append(token2index.get(char, token2index.get("<unk>")))
        return torch.tensor([number_data], dtype=torch.long), torch.tensor([len(number_data)], dtype=torch.long)

    token2index, index2tag = _load_dict(param.save_dir)
    param.vocab_size = len(token2index)
    param.num_classes = len(index2tag)
    model = ESIM_PL.load_from_checkpoint(param.pre_ckpt_path, parm=param)
    test_data = {"sentence1": "杭州哪里好玩", "sentence2": "杭州哪里好玩点"}
    result_index = \
    model.forward(*_number_data(test_data["sentence1"]), *_number_data(test_data["sentence2"]))[1].argmax(dim=-1)[0].item()
    print(index2tag.get(str(result_index)))  # 1


if __name__ == '__main__':
    model_parser = ESIM.add_argparse_args()
    parser = ArgumentParser(parents=[model_parser])
    parser.add_argument('-lr', type=float, default=5e-3, help='学习率')
    parser.add_argument('-data_dir', type=str, default="corpus/LCQMC_S", help='训练语料地址')
    parser.add_argument('-batch_size', type=int, default=300, help='批次数据大小')
    parser.add_argument('-epoch', type=int, default=15)
    parser.add_argument('-embedding_dim', type=int, default=60, help='词向量的维度')
    parser.add_argument('-hidden_size', type=int, default=128, help='lstm 隐层神经元数')
    parser.add_argument('-save_dir', type=str, default="model_save/esim", help='模型存储位置')
    parser.add_argument('-load_pre', type=bool, default=False, help='是否加载已经训练好的ckpt')
    parser.add_argument('-pre_ckpt_path', type=str,
                        default="model_save/esim/esim-epoch=002-val_loss=0.416-f1_score=0.829.ckpt",
                        help='是否加载已经训练好的ckpt')

    args = parser.parse_args()
    training(args)
    # model_use(args)

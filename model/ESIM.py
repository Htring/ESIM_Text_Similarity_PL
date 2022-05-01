#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: ESIM.py
@time:2022/04/23
@description:
"""
from argparse import Namespace, ArgumentParser
import torch
from torch import nn
from .layers import RNNEncoder, SoftmaxAttention
from .utils import get_mask, replace_masked
import torch.nn.functional as F


class ESIM(nn.Module):
    __doc__ = """ ESIM模块 """

    @staticmethod
    def add_argparse_args() -> ArgumentParser:
        parser = ArgumentParser(description='esim', add_help=False)
        parser.add_argument('-dropout', type=float, default=0.5, help='dropout层参数， default 0.5')
        return parser

    def __init__(self, parm: Namespace):
        super().__init__()
        self.hidden_size = parm.hidden_size
        self.num_classes = parm.num_classes
        self.embedding = nn.Embedding(parm.vocab_size, parm.embedding_dim)
        self.embedding_dim  = parm.embedding_dim
        self.dropout_layer = nn.Dropout(parm.dropout)
        self.rnn_encoder = RNNEncoder(rnn_type=nn.LSTM,
                                      input_size=self.embedding_dim,
                                      hidden_size=self.hidden_size,
                                      bidirectional=True)
        self.attention = SoftmaxAttention()

        self.projection = nn.Sequential(nn.Linear(4*2*self.hidden_size, self.hidden_size),
                                        nn.ReLU())

        self.composition = RNNEncoder(nn.LSTM,
                                      input_size=self.hidden_size,
                                      hidden_size=self.hidden_size,
                                      bidirectional=True)
        self.classification = nn.Sequential(self.dropout_layer,
                                            nn.Linear(2*4*self.hidden_size, self.hidden_size),
                                            nn.Tanh(),
                                            self.dropout_layer,
                                            nn.Linear(self.hidden_size, self.num_classes))

    def forward(self, premises: torch.Tensor, premises_lengths: torch.Tensor, hypotheses: torch.Tensor,
                hypotheses_lengths: torch.Tensor):
        """
        :param premises: A batch of variable length sequences of word indices representing premises. The batch is
                         assumed to be of size (batch, premises_length).
        :param premises_lengths: A 1D tensor containing the lengths of the premises in 'premises'.
        :param hypotheses: A batch of variable length sequences of word indices representing hypotheses. The batch is
                           assumed to be of size (batch, hypotheses_length).
        :param hypotheses_lengths: A 1D tensor containing the lengths of the hypotheses in 'hypotheses'.
        :return: logits: A tensor of size (batch, num_classes) containing the logits for each output class of the model.
                 probabilities: A tensor of size (batch, num_classes) containing the probabilities of each output class in the model.
        """
        device = premises.device
        embedded_premises = self.embedding(premises)
        embedded_hypotheses = self.embedding(hypotheses)
        premises_mask = get_mask(premises, premises_lengths).to(device)
        hypotheses_mask = get_mask(hypotheses, hypotheses_lengths).to(device)

        encoded_premises = self.rnn_encoder(embedded_premises, premises_lengths)
        encoded_hypotheses = self.rnn_encoder(embedded_hypotheses, hypotheses_lengths)

        attended_premises, attended_hypotheses = self.attention(encoded_premises, premises_mask,
                                                                encoded_hypotheses, hypotheses_mask)
        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses - attended_hypotheses,
                                         encoded_hypotheses * attended_hypotheses],
                                        dim=-1)
        projected_premises = self.projection(enhanced_premises)
        projected_hypotheses = self.projection(enhanced_hypotheses)
        v_ai = self.composition(projected_premises, premises_lengths)
        v_bj = self.composition(projected_hypotheses, hypotheses_lengths)
        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1).transpose(2, 1), dim=1) / torch.sum(premises_mask, dim=1,
                                                                                                  keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1).transpose(2, 1), dim=1) / torch.sum(hypotheses_mask,
                                                                                                    dim=1, keepdim=True)
        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)
        logits = self.classification(v)
        probabilities = F.softmax(logits, dim=-1)
        return logits, probabilities

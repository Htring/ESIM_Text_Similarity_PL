#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: utils.py
@time:2022/04/24
@description: 模型训练工具
"""

import torch
import torch.nn.functional as F


def sort_by_seq_lens(batch: torch.Tensor, sequences_lengths: torch.Tensor, descending: bool = True):
    """
    Sort a batch of padded variable length sequences by their length.
    :param batch: A batch of padded variable length sequences. The batch should have the dimensions
                  (batch_size x max_sequence_length x *).
    :param sequences_lengths: A tensor containing the lengths of the sequences in the input batch.
                              The tensor should be of size (batch_size).
    :param descending: A boolean value indicating whether to sort the sequences by their lengths in descending order.
                       Defaults to True.
    :return:
      sorted_batch: A tensor containing the input batch reordered by sequences lengths.
      sorted_seq_lens: A tensor containing the sorted lengths of the sequences in the input batch.
      sorting_idx: A tensor containing the indices used to permute the input batch in order to get 'sorted_batch'.
      restoration_idx: A tensor containing the indices that can be used to restore the order of the sequences in
                       'sorted_batch' so that it matches the input batch.
    """
    sorted_seq_lens, sorting_index = sequences_lengths.sort(0, descending=descending)
    sorted_batch = batch.index_select(0, sorting_index)
    idx_range = torch.arange(0, len(batch)).type_as(sequences_lengths)
    _, reverse_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, reverse_mapping)
    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index


def get_mask(sequences_batch: torch.Tensor, sequences_lengths: torch.Tensor):
    """
    Get the mask for a batch of padded variable length sequences.
    pad index number must equal to 0
    :param sequences_batch: A batch of padded variable length sequences containing word indices. Must be a 2-dimensional
                            tensor of size (batch, sequence).
    :param sequences_lengths: A tensor containing the lengths of the sequences in 'sequences_batch'. Must be of size
                             (batch).
    :return: A mask of size (batch, max_sequence_length), where max_sequence_length is the length of the longest
             sequence in the batch.
    """
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones([batch_size, max_length], dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = 0
    return mask


def masked_softmax(batch: torch.Tensor, mask: torch.Tensor):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    :param batch: The tensor on which the softmax function must be applied along the last dimension.
    :param mask: A mask of the same size as the tensor with 0s in the positions of
                 the values that must be masked and 1s everywhere else.
    :return: A tensor of the same size as the inputs containing the result of the softmax.
    """
    batch_shape = batch.size()
    reshaped_batch = batch.view(-1, batch_shape[-1])
    while mask.dim() < batch.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(batch).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])
    result = F.softmax(reshaped_mask * reshaped_batch, dim=-1)
    # result = result * reshaped_mask
    # result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(batch_shape)


def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.
    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.
    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)
    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()
    return weighted_sum * mask


# Code inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def replace_masked(tensor, mask, value):
    """
    Replace the all the values of vectors in 'tensor' that are masked in 'masked' by 'value'.
    Args:
        tensor: The tensor in which the masked vectors must have their values replaced.
        mask: A mask indicating the vectors which must have their values replaced.
        value: The value to place in the masked vectors of 'tensor'.
    Returns:
        A new tensor of the same size as 'tensor' where the values of the vectors masked in 'mask' were replaced by 'value'.
    """
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add


# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class PointerNet(nn.Module):
    def __init__(self, query_vec_size, src_encoding_size, attention_type='multi_options'):
        super(PointerNet, self).__init__()

        assert attention_type in ('affine', 'dot_prod', 'multihead', 'multi_options')

        if attention_type == 'affine':
            self.src_encoding_linear = nn.Linear(src_encoding_size, query_vec_size, bias=False)
        if attention_type == 'multihead':
            self.numheads = 4
            # values
            self.src_encoding_linear = nn.Linear(src_encoding_size, query_vec_size * self.numheads, bias=False)
            self.multihead_combiner = nn.Linear(self.numheads, 1, bias=False)
        if attention_type == 'multi_options':
            self.numheads = 4
            # values
            self.src_encoding_linear = nn.Linear(src_encoding_size, query_vec_size * self.numheads, bias=False)
            # this will give us a choice between which head to use to choose
            self.options_linear = nn.Linear(query_vec_size, self.numheads, bias=False)

        self.attention_type = attention_type

    def forward(self, src_encodings, src_token_mask, query_vec):
        """
        :param src_encodings: Variable(batch_size, src_sent_len, hidden_size * 2)
        :param src_token_mask: Variable(batch_size, src_sent_len)
        :param query_vec: Variable(tgt_action_num, batch_size, query_vec_size)
        :return: Variable(tgt_action_num, batch_size, src_sent_len)
        """

        # (batch_size, 1, src_sent_len, query_vec_size (or numheads times))
        if self.attention_type in ['affine', 'multihead', 'multi_options']:
            src_encodings = self.src_encoding_linear(src_encodings)  # this happens
        src_encodings = src_encodings.unsqueeze(1)

        # (batch_size, tgt_action_num, query_vec_size, 1)
        q = query_vec.permute(1, 0, 2).unsqueeze(3)

        if self.attention_type in ['multihead', 'multi_options']:
            src_encodings = F.leaky_relu(src_encodings)  # otherwise whats the difference?
            batch_size, _, src_sent_len, query_vec_size = src_encodings.shape  # well, query_vec_size is numheads times bigger than actual
            # batch_size, tgt_action_num, src_sent_len, numheads, something
            src_encodings = src_encodings.contiguous().view(batch_size, _, src_sent_len,
                                                            self.numheads, query_vec_size // self.numheads)

            # (batch_size, tgt_action_num, src_sent_len, numheads)
            weights = torch.matmul(src_encodings, q.unsqueeze(-3)).squeeze(-1)

            # (tgt_action_num, batch_size, src_sent_len, numheads)
            weights = weights.permute(1, 0, 2, 3)

            if self.attention_type == "multi_options":
                # (tgt_action_len, batch_size, numheads)
                options_scores = self.options_linear(query_vec)  # Try softmax?
                # (tgt_action_len, batch_size, 1), values are index to options
                best_options = torch.argmax(options_scores, -1, keepdim=True)
                best_options = best_options.unsqueeze(2).repeat([1, 1, weights.shape[2], 1])
                weights = weights.gather(dim=-1, index=best_options).squeeze(-1)
            else:  # take a combination
                weights = self.multihead_combiner(weights).squeeze(-1)

            if src_token_mask is not None:
                # (tgt_action_num, batch_size, src_sent_len)
                src_token_mask = src_token_mask.unsqueeze(0).expand_as(weights)
                # (tgt_action_num, batch_size, src_sent_len)
                weights.data.masked_fill_(src_token_mask, -float('inf'))

        else:
            # (batch_size, tgt_action_num, src_sent_len)
            weights = torch.matmul(src_encodings, q).squeeze(3)

            # (tgt_action_num, batch_size, src_sent_len)
            weights = weights.permute(1, 0, 2)

            if src_token_mask is not None:
                # (tgt_action_num, batch_size, src_sent_len)
                src_token_mask = src_token_mask.unsqueeze(0).expand_as(weights)
                weights.data.masked_fill_(src_token_mask, -float('inf'))

        ptr_weights = F.softmax(weights, dim=-1)  # tgt_action_num, batch_size, src_sent_len

        return ptr_weights

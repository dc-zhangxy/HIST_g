# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from typing import Text, Union
import copy
import math
from qlib.utils import get_or_create_path
from qlib.log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim

from qlib.model.base import Model 
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class Transformer(nn.Module):
    def __init__(self, d_feat=6, d_model=6, nhead=3, num_layers=2, dropout=0.1, device=None):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # self.decoder_layer = nn.Linear(d_model, 4) # 1
        # self.pos_decoder = PositionalEncoding(d_model)
        # self.label_layer = nn.Linear(d_feat, d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        
        self.predictor = nn.Linear(d_model, d_feat)
        
        self.device = device
        self.d_feat = d_feat

    def forward(self, src, tgt_input):
        # src [N, F*T] --> [N, T, F]
        src = src.reshape(len(src), self.d_feat, -1).permute(0, 2, 1)
        src = self.feature_layer(src)

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first
        
        src = self.pos_encoder(src)
        enc_outputs = self.transformer_encoder(src, mask=None)  # [T, N, F] [60, 512, 8] # mask = None
        
        enc_outputs = enc_outputs[-1:, :, :] # 取最后一天的输出 enc_output.transpose(1, 0)[:, -1:, :]

        tgt = tgt_input[:1,:,:]
        dec_outputs = self.transformer_decoder(tgt, enc_outputs)  
        predict = self.predictor(dec_outputs[:1, : ,:])
        for i in range(1,5):
            tgt = torch.concat([tgt, predict], dim=0)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(i+1).to(self.device)
            dec_outputs = self.transformer_decoder(tgt, enc_outputs, tgt_mask=tgt_mask)  
            predict = self.predictor(dec_outputs[i:i+1, : ,:])
            
        # [day+1, batch, feature]
        return tgt.transpose(1, 0)  #permute(1, 0, 2) .squeeze()

 # train 用真实值train，但test用预测值test       
def test(model, enc_input, start_symbol=torch.zeros(5)):
    tgt_len = 5
    # Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    # enc_outputs, enc_self_attns = model.Encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model(enc_input, dec_input)
        projected = model.projection(dec_outputs)
        # prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        # next_word = prob.data[i]
        next_symbol = projected # next_word.item() #dec_outputs[i]
    return dec_input

def test2(model, src, tgt):
    # 一个一个词预测，直到预测为<eos>，或者达到句子最大长度
    for i in range(5):
        # 进行transformer计算
        out = model(src, tgt)
        # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
        predict = model.predictor(out[:, -1])

        # 和之前的预测结果拼接到一起
        tgt = torch.concat([tgt, predict.unsqueeze(0)], dim=1)

    print(tgt)
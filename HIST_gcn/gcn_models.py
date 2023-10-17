import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GCN(nn.Module):
    def __init__(self, nfeat=64, nhid=64, nclass=64, dropout=0.05):
        super(GCN, self).__init__()

        self.gru_model = GRUModel()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU()

        self.fc_hs_fore = nn.Linear(64, 64)
        torch.nn.init.xavier_uniform_(self.fc_hs_fore.weight)

        self.fc_hs_back = nn.Linear(64, 64)
        torch.nn.init.xavier_uniform_(self.fc_hs_back.weight)

        self.fc_out = nn.Linear(nclass, 1)
        self.fc_indi = nn.Linear(64, 64)
        torch.nn.init.xavier_uniform_(self.fc_indi.weight)

    def forward(self, x, adj):
        x_hidden = self.gru_model(x)

        h_shared_info = x_hidden 
        x = F.relu(self.gc1(h_shared_info, adj.float()))  #  .to(torch.float32)
        x = F.dropout(x, self.dropout, training=self.training) ## 
        x = self.gc2(x, adj.float() )  #  .to(torch.float32)
        # out = F.log_softmax(x, dim=1)
        h_shared_back = self.fc_hs_back(x)
        output_hs = self.fc_hs_fore(x)
        output_hs = self.leaky_relu(output_hs)
        
        # 加一个残差项
        individual_info  = x_hidden  - h_shared_back
        output_indi = individual_info
        output_indi = self.fc_indi(output_indi)
        output_indi = self.leaky_relu(output_indi)
        all_info = output_hs + output_indi
        pred_all = self.fc_out(all_info).squeeze()
        
        return pred_all.to(torch.float64)


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input0, adj):
        support = torch.mm(input0, self.weight)
        # output = torch.spmm(adj, support)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GRUModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, gru_output=64 ,dropout=0.0):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, gru_output)

        self.d_feat = d_feat
        self.day_long = 30

    def forward(self, x):
        # print(x.dtype)
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        # x = x.reshape(len(x), self.day_long, self.d_feat)  # [N, T, F]
        # x = torch.flip(x, dims=[1])  # 按时间由远至近
        out, _ = self.rnn(x.float())  ### 
        return self.fc_out(out[:, -1, :]) #.squeeze()

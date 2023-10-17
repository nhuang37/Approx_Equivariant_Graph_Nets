from __future__ import absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.sem_graph_conv_triv import SemGraphConv
from models.graph_non_local import GraphNonLocal


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None, gc_flag=True, ew_flag=True, pt=False):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj, gc_flag=gc_flag, ew_flag=ew_flag, pt=pt)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout,  gc_flag=True, ew_flag=True, pt=False):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout, gc_flag=gc_flag, ew_flag=ew_flag, pt=pt)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout, gc_flag=gc_flag, ew_flag=ew_flag, pt=pt)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out



class SemGCNtriv(nn.Module):
    '''
    Perform the irreducible decomposition back and forth!
    '''
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, p_dropout=None, gc_flag=True, ew_flag=True, pt=False):
        super(SemGCNtriv, self).__init__()
        _gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout,  
                                    gc_flag=gc_flag, ew_flag=ew_flag, pt=pt)]
        _gconv_layers = []

        for i in range(num_layers):
            _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout,  
                               gc_flag=gc_flag, ew_flag=ew_flag, pt=pt))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = SemGraphConv(hid_dim, coords_dim[1], adj,  gc_flag=gc_flag, ew_flag=ew_flag, pt=pt)

    def forward(self, x):
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        out = self.gconv_output(out)
        return out
from __future__ import absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.sem_graph_conv_autG import SemGraphConv

from models.graph_non_local import GraphNonLocal


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, pairs, singletons, basis, p_dropout=None, gc_flag=True, ew_flag=True,
                  global_flip='NA', pt=False, blocks=[0,10,13,16]):
        super(_GraphConv, self).__init__()
        if global_flip == 'global':
            self.gconv =  SemGCflip(input_dim, output_dim, adj, basis, gc_flag, ew_flag, pt) #use (Z/2Z) only
        elif global_flip == 'orbit':
            self.gconv =  SemGCob(input_dim, output_dim, adj, pairs, singletons, basis, gc_flag, ew_flag, pt) #use (Z/2Z)^6
        else:
            self.gconv = SemGraphConv(input_dim, output_dim, adj, basis, gc_flag=gc_flag, ew_flag=ew_flag, pt=pt, blocks=blocks) #(Z/2Z)^2
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
    def __init__(self, adj, input_dim, output_dim, hid_dim, pairs, singletons, basis, p_dropout, gc_flag=True, ew_flag=True, 
                 global_flip='NA', pt=False, blocks=[0,10,13,16]):
        super(_ResGraphConv, self).__init__()
        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, pairs, singletons, basis, p_dropout, gc_flag, ew_flag, global_flip, pt, blocks)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, pairs, singletons, basis, p_dropout, gc_flag, ew_flag, global_flip, pt, blocks)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out



class SemGCNautG(nn.Module):
    '''
    Perform the irreducible decomposition back and forth!
    '''
    def __init__(self, adj, hid_dim, pairs, singletons, coords_dim=(2, 3), num_layers=4, p_dropout=None,
                 gc_flag=True, ew_flag=True, global_flip='NA', pt=False, blocks=[0,10,13,16]):
        super(SemGCNautG, self).__init__()
        self.singletons = singletons
        self.pairs = pairs #store the node pairs
        self.B_inv = []
        self.B_flip = []
        self.E = torch.eye(16) #e_0, ..., e_15

        for (left, right) in self.pairs:
          self.B_inv.append( (1/np.sqrt(2))*(self.E[:,left] + self.E[:, right]) )
          self.B_flip.append( (1/np.sqrt(2))*(self.E[:,left] - self.E[:, right]) )
        for node in self.singletons:
          self.B_inv.append(self.E[:,node])
        assert len(self.B_inv) == 10, "invariant subspace is not computed correctly!"
        assert len(self.B_flip) == 6, "flip subspace is not computed correctly!"
        self.B_inv = torch.stack(self.B_inv).T
        self.B_flip = torch.stack(self.B_flip).T
        #the isotypic representation basis
        self.B = torch.cat([self.B_inv, self.B_flip], dim=1)
        
        assert torch.allclose(self.B @ self.B.T, torch.eye(16), rtol=1e-5, atol=1e-5)
        
        _gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, pairs, singletons, self.B, p_dropout, gc_flag, ew_flag, global_flip, pt, blocks)]
        
        _gconv_layers = []

        for i in range(num_layers):
            _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, pairs, singletons, self.B, 
                                               p_dropout, gc_flag, ew_flag, global_flip, pt, blocks))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        if global_flip == 'global':
            self.gconv_output = SemGCflip(hid_dim, coords_dim[1], adj, self.B, p_dropout, gc_flag, ew_flag, pt)
        elif global_flip == 'orbit':
            self.gconv_output = SemGCob(hid_dim, coords_dim[1], adj, self.pairs, self.singletons,
                                        self.B, p_dropout, gc_flag, ew_flag, pt)
        else:
            self.gconv_output = SemGraphConv(hid_dim, coords_dim[1], adj, self.B, gc_flag=gc_flag, ew_flag=ew_flag, pt=pt, blocks=blocks)

    def forward(self, x):
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        out = self.gconv_output(out)
        return out
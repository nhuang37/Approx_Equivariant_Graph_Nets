from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

###parameterize all linear equivariant function w.r.t aut(G) = Z/2Z

###aut(G) = I (trivial group, enforce no equivariance whatsoever)

class SemGraphConv(nn.Module):
    """
    Semantic graph convolution layer that enforces no equivariance!
    Input: batch_size x 16 x in_features
    Output: batch_size x 16 x out_features
    Arbitrary linear function f: 16 x in_feat -> 16 x out_feat; learn a big fat linear layer
    """

    def __init__(self, in_features, out_features, adj, bias=True, gc_flag=True, ew_flag=True, pt=False):
        super(SemGraphConv, self).__init__()
        self.n = adj.shape[0]
        self.in_features = in_features
        self.out_features = out_features
        self.gc_flag = gc_flag #whether to apply graph convolution A; default True
        self.ew_flag = ew_flag #whether to learn edge weights M; default True
        self.pt = pt #whether to apply graph A as pointwise multiplying f equivariant map; default False

        #d x d' linear equivariant blocks
        self.W = nn.Parameter(torch.rand(self.n*self.in_features, self.n*self.out_features)) 
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.out_features)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

        #weighted graph (semGCN design, keep)
        self.adj = adj
        self.m = (self.adj > 0)
        if self.ew_flag or self.pt: #pt(ew)
            self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
            nn.init.constant_(self.e.data, 1)

        self.reset_weights()

    def reset_weights(self):
        #default nn.linear init: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L105
        stdv = 1. / math.sqrt(self.n*self.in_features)
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, input):
        ## linear block
        batch_size = input.shape[0]
        self.m = self.m.to(input.device)

        if self.pt: #(A \odot f) x 
            adj = -1 * torch.ones_like(self.adj).to(input.device) #-9e15 (basically set non-edge to zero)
            adj[self.m] = self.e #update the weights at each forward pass
            adj = F.softmax(adj, dim=1)
            f_sp = torch.tile(adj, (self.in_features, self.out_features)) * self.W #(out*n, in*n)

            #the pointwise mask is arranged in column-wise major order (e1, e2, ..., f1, f2...)
            #so we permute the input axes accordingly to follow (e1, e2, ..., f1, f2)
            h = F.linear(input.permute(0,2,1).reshape(batch_size, -1), f_sp.T)#self.W(input.reshape(batch_size, -1))
            h = h.reshape(batch_size, -1, self.n).permute(0,2,1) #permute back
        else:
            h = F.linear(input.reshape(batch_size, -1), self.W.T)#self.W(input.reshape(batch_size, -1))
            h = h.reshape(batch_size, self.n, -1)

        if self.gc_flag and self.ew_flag:
            assert self.pt == False, "cannot have both gc and pointwise on!"
            #this is spatial again: using standard basis
            adj = -9e15 * torch.ones_like(self.adj).to(input.device)
            adj[self.m] = self.e #update the weights at each forward pass
            adj = F.softmax(adj, dim=1)
            output = torch.matmul(adj.to(input.device), h)
        elif self.gc_flag:
            output = torch.matmul(self.adj.to(input.device), h)
        else:
            output = h

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output
        
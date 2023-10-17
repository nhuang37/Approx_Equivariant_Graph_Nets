from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemGraphConv(nn.Module):
    """
    S_16 equivariant graph layer (c.f. deepset / point net)
    encompasing original SemGraphConv by using gc_flag = True; ew_flag = True
    """

    def __init__(self, in_features, out_features, adj, bias=True, gc_flag=True,
                 ew_flag=True, pt=False, tie_all=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gc_flag = gc_flag 
        self.ew_flag = ew_flag
        self.pt = pt #pointwise graph multiply
        self.tie_all = tie_all #tie_all: S_16, otherwise, Relax S_16
        self.n = adj.shape[0]

        if tie_all: #2 scalars
            self.w_diag = nn.Parameter(torch.rand(self.in_features, self.out_features)) 
            self.w_off = nn.Parameter(torch.rand(self.in_features, self.out_features))
        else:
        #16 nodes -> 32 scalars
            self.w_diag = nn.Parameter(torch.rand(self.in_features, self.out_features, self.n)) 
            self.w_off = nn.Parameter(torch.rand(self.in_features, self.out_features, self.n))

        self.adj = adj
        self.m = (self.adj > 0)
        if ew_flag or self.pt:
            self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
            nn.init.constant_(self.e.data, 1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.out_features)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

        self.reset_weights()

    def reset_weights(self):
        stdv = 1. / math.sqrt(self.in_features * self.n) 
        self.w_diag.data.uniform_(-stdv, stdv)
        self.w_off.data.uniform_(-stdv, stdv)


    def forward(self, input): #input: (bs, n, d_in)
        batch_size = input.shape[0]
        M = torch.eye(self.adj.size(0), dtype=torch.float).to(input.device)

        if self.tie_all:
            W_e = self.w_diag.reshape(self.in_features, self.out_features,1, 1).expand(-1, -1, self.n, self.n) #(self.in_features, self.out_features, n,n, )
            W_f = self.w_off.reshape(self.in_features, self.out_features,1, 1).expand(-1, -1, self.n, self.n) #(self.in_features, self.out_features, n,n, )
        else: #per node
            W_e = self.w_diag.unsqueeze(-1).expand(-1, -1, -1, self.n)
            W_f = self.w_off.unsqueeze(-1).expand(-1, -1, -1, self.n) 

        W = M.mul(W_e) + (1-M).mul(W_f) # (self.in_features, self.out_features, n,n,)
        if self.pt:  #A \odot f$
            assert self.gc_flag == False, 'cannot have both gc or pt'
            adj = -1 * torch.ones_like(self.adj).to(input.device) #-9e15 (basically set non-edge to zero)
            adj[self.m] = self.e #update the weights at each forward pass
            adj = F.softmax(adj, dim=1)  
            W = adj.mul(W) #(d_in, d_out, n,n, )

        W = W.permute(1,0, 2, 3) #(d_out, d_in , n, n)
        W = W.flatten(start_dim=1, end_dim=2) #(d_out, d_in * n, n) #so far this is good
        input = input.permute(0,2,1).reshape(batch_size, -1) #to match W with the order (e1, e2, ... f1, f2...)
        h = torch.matmul(input, W) #fixing #(d_out, bs, n)
        h = h.permute(1,2,0) #(batch, n, d_out)

        if self.gc_flag and self.ew_flag:
            assert self.pt == False, "cannot have both gc and pointwise on!"
            #this is spatial again: using standard basis
            adj = -9e15 * torch.ones_like(self.adj).to(input.device)
            adj[self.m] = self.e #update the weights at each forward pass
            adj = F.softmax(adj, dim=1)
            M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
            output = torch.matmul(adj * M, h)
        elif self.gc_flag:
            output = torch.matmul(self.adj.to(input.device), h)
        else:
            output = h
        
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

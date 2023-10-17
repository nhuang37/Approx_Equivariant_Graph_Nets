from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

###parameterize all linear equivariant function w.r.t aut(G) = (Z/2Z)^2

###aut(G): (Z/2Z)^2

def indexer(n, d, blocks):
    '''
    return the correct order of irrep basis after block_diag B
    blocks: a list contains block partition index (e.g., [0, 10, 13, 16]; [0,1,2])
    '''
    assert blocks[-1] == n, "must add up to n"
    ind = np.arange(n*d).reshape(d,n)
    indices = []
    for start, end in zip(blocks[:-1], blocks[1:]):
        ind_sub = ind[:, start:end]
        #print(ind_sub)
        indices += ind_sub.flatten().tolist()
    assert len(indices) == n*d, "not right yet!"
    return indices

class SemGraphConv(nn.Module):
    """
    Semantic graph convolution layer that preserves (Z/2Z)^2 equivariance in the human skeleton graph
    Use the equivariant basis B (i.e., isotypic representation) for given input 
    Parameterize linear equivariant functions using B => parameterize linear functions in each irreducible block followed by disjoint union
    Final output: project the hidden embedding in the equivariant basis B back to the standard basis 
    """

    def __init__(self, in_features, out_features, adj, basis, bias=True, gc_flag=True, ew_flag=True, pt=False, blocks=[0,10,13,16]):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.B = basis
        self.gc_flag = gc_flag #whether to apply graph convolution A; default True
        self.ew_flag = ew_flag #whether to learn edge weights M; default True
        self.pt = pt #whether to apply graph A as pointwise multiplying f equivariant map; default False
        self.blocks = blocks #irrep block partition (list)
        self.n = adj.shape[0]
        #d x d' linear equivariant blocks: 
        ##No block-wise bias is allowed otherwise equivariance breaks
        self.W_e = nn.Linear(self.in_features*10, self.out_features*10, bias=False)
        self.W_2 = nn.Linear(self.in_features*3, self.out_features*3, bias=False)
        self.W_3 = nn.Linear(self.in_features*3, self.out_features*3, bias=False)
     
        #overall bias for everyone (16 nodes)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.out_features)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

        #weighted graph (semGCN design, keep)
        self.adj = adj
        self.m = (self.adj > 0)
        if self.ew_flag:
            self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
            nn.init.constant_(self.e.data, 1)

        if self.pt:
            #blow-up basis
            ind_in = indexer(self.n, self.in_features, self.blocks)
            ind_out =  indexer(self.n, self.out_features, self.blocks)
            self.bas_in = torch.block_diag(*(self.B.T.expand(self.in_features, -1, -1)))[ind_in] #big bug, basis now it is B.T think of f(x) = f x multiplying from the left
            self.bas_out = torch.block_diag(*(self.B.expand(self.out_features,-1,-1)))[ind_out]
            self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
            nn.init.constant_(self.e.data, 1)

    def forward(self, input):
        self.B = self.B.to(input.device)
        self.m = self.m.to(input.device)

        #input: batch_size, n, dim
        batch_size = input.shape[0]
        if self.pt: #(A \odot f) x = diag ( A Diag(x) f^T ) = diag (A Diag(x) B W^T B^T)
            self.bas_in = self.bas_in.to(input.device)
            self.bas_out = self.bas_out.to(input.device)
            W = torch.block_diag(self.W_e.weight, self.W_2.weight, self.W_3.weight) #fast hack (in_feat x 16, out_feat x 16)
            #get our equivariant map: (out*n, in*n) 
            f = self.bas_out @ W @ self.bas_in 
            adj = -1 * torch.ones_like(self.adj).to(input.device) #-9e15 (basically set non-edge to zero)
            adj[self.m] = self.e #update the weights at each forward pass
            adj = F.softmax(adj, dim=1)
            f_sp = torch.tile(adj, (self.out_features, self.in_features)) * f #(out*n, in*n)
            #act on the input: we must permute, because our irrep basis maps all d_in inv basis first, ...
            input = input.permute(0,2,1).reshape(batch_size, -1) # (batch_size, dim , n) => (batch_size, dim * n ) # this interleaves w/ f.reshape!
            h = F.linear(input, f_sp) #(batch_size, out_features*n)
            h = h.reshape(batch_size, self.out_features, self.n).permute(0,2,1) #(batch_size, n, dim)

        else:
            #project to irrep basis: bmm with broacasting with (dim, batch_size, n) (note: must use permute instead of reshape!)
            input_irrep = torch.matmul(input.permute(2,0,1), self.B) #note: for batch of vectors, we right multiply by B instead to perform change of basis
            input_irrep = input_irrep.permute(1,0,2) #batch_size, dim, n
            #linear equivariant layer 
            out_e = self.W_e(input_irrep[:,:,:10].reshape(batch_size, -1))
            out_e = out_e.reshape(batch_size, -1, 10)
            out_2 = self.W_2(input_irrep[:,:,10:13].reshape(batch_size, -1))
            out_2 = out_2.reshape(batch_size, -1, 3)
            out_3 = self.W_3(input_irrep[:,:,13:].reshape(batch_size, -1))
            out_3 = out_3.reshape(batch_size, -1, 3)
            out_irrep = torch.cat([out_e, out_2, out_3], dim=2) # batch_size, out_features, n

            #project back to standard basis
            h = torch.matmul(out_irrep, self.B.T)
            h = h.permute(0,2,1) #batch_size, n, out_features
        #print(h.shape, self.bias.data.shape)

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
from __future__ import absolute_import, division


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch_geometric
from torch.utils.data import Subset, Dataset
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import networkx as nx
import pickle
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import math
from einops import rearrange

##HELPER FUNCTIONS

def flatten_ids(ids):
    return [c for cluster in ids for c in cluster]

def inv(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


class Sn_coarsen_layer(nn.Module):
    def __init__(self, cluster_ids, in_features, out_features, N=28):
        '''
        Assume 1-layer linear model 
        Rewrite for simple 1d case
        Generalize to d_in, d_out
        '''
        super(Sn_coarsen_layer, self).__init__()
        self.cluster_ids = cluster_ids
        self.permute_ids = flatten_ids(cluster_ids) #rearrange nodes
        self.inv_permute = inv(self.permute_ids) #inverse permutation
        self.num_cluster = len(cluster_ids)
        self.num_nodes = len(self.permute_ids)
        self.cluster_sizes = np.array([len(c) for c in cluster_ids])
        self.cut_offs = [0] + list(np.cumsum(self.cluster_sizes)) #cut off at cluster size
        #print(self.cut_offs)

        self.ratio = len(self.cluster_ids[0]) #cluster size
        self.in_features = in_features
        self.out_features = out_features
        self.w_diag = nn.Parameter(torch.rand(self.in_features * self.out_features, self.num_cluster))
        self.w_off = nn.Parameter(torch.rand(self.in_features * self.out_features, self.num_cluster, self.num_cluster)) #TODO- rand vs ones?
              
        self.b1 = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
        stdv = 1. / math.sqrt(out_features)
        self.b1.data.uniform_(-stdv, stdv)
        #TODO: might be fun to check the learned cluster message matrix!
        #print("Warning: this asserts patched-clustering for regular grid images!")

        #self.reset_weight()
    
    def reset_weight(self):
        stdv = 1. / math.sqrt(self.out_features) #potential big game changer! (before forget self.n)
        self.w_diag.data.uniform_(-stdv, stdv)
        self.w_off.data.uniform_(-stdv, stdv) #smaller, proportional

    def forward(self, x):
        #layer 1 updates
        bs, num_nodes, in_features = x.shape
        if self.num_cluster < num_nodes:
            M_diag = torch.zeros((self.in_features*self.out_features, num_nodes, num_nodes)).to(x.device) #placeholder
            M_off = torch.zeros((self.in_features*self.out_features, num_nodes, num_nodes)).to(x.device) #placeholder
            for i in range(self.num_cluster):
                block_diag = self.w_diag[:, i].unsqueeze(-1).unsqueeze(-1).expand(-1, self.cluster_sizes[i], self.cluster_sizes[i])
                M_diag[:, self.cut_offs[i]:self.cut_offs[i+1], self.cut_offs[i]:self.cut_offs[i+1] ] = block_diag

                for j in range(self.num_cluster):
                    block_off = self.w_off[:, i,j].unsqueeze(-1).unsqueeze(-1).expand(-1, self.cluster_sizes[i], self.cluster_sizes[j])
                    M_off[:, self.cut_offs[i]:self.cut_offs[i+1], self.cut_offs[j]:self.cut_offs[j+1] ] = block_off

            #first make the diagonal matrix, then broadcast by kron, then rearrange to tile
            W_diag = rearrange(M_diag, '(i1 i2) j k -> (i1 j) (i2 k)', i1=self.out_features) #
            W_off = (1/ self.num_nodes) * rearrange(M_off, '(i1 i2) j k -> (i1 j) (i2 k)', i1=self.out_features)
        else: #triv! no ops needed
            W_diag = rearrange(torch.diag_embed(self.w_diag), '(i1 i2) j k -> (i1 j) (i2 k)', i1=self.out_features)
            W_off = (1/ self.num_nodes) * rearrange(self.w_off, '(i1 i2) j k -> (i1 j) (i2 k)', i1=self.out_features)

        W = W_diag + W_off #(d_out*N, d_in*N)

        #rearange nodes according to cluster membership
        x = x[:, self.permute_ids, :]  #(bs, N, d)
        x = x.permute(0,2,1).reshape(bs, -1) #flatten to have [N; N; N ...]
        #linear map
        out = F.linear(x, W) #(bs, d_out * N)
        out = out.reshape(bs, self.out_features, num_nodes).permute(0,2,1) #(bs, N, d_out)
        #rearange to original node ordering
        out = out[:, self.inv_permute, :]
        out += self.b1.view(1, 1, -1)
        return out
    
class Sn_coarsen_net(nn.Module):
    def __init__(self, cluster_ids, in_features, out_features, hid_dim):
        super(Sn_coarsen_net, self).__init__()
        self.cluster_ids = cluster_ids
        self.permute_ids = flatten_ids(cluster_ids) #rearrange nodes
        self.inv_permute = inv(self.permute_ids) #inverse permutation
        self.num_cluster = len(cluster_ids)
        self.num_nodes = len(self.permute_ids)
        self.ratio = len(self.cluster_ids[0]) #cluster size
        #layers
        self.input_layer = Sn_coarsen_layer(cluster_ids, in_features, hid_dim)
        self.output_layer = Sn_coarsen_layer(cluster_ids, hid_dim, out_features)
        self.reset_weight()

    def reset_weight(self):
        for layer in [self.input_layer, self.output_layer]:
            layer.reset_weight()    
    
    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.output_layer(x)

        return x
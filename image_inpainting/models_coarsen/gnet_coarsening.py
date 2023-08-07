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
from models_coarsen.gnet_coarsen_utils import reflection_clusters

##HELPER FUNCTIONS

def flatten_ids(ids):
    return [c for cluster in ids for c in cluster]

def inv(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

def swap_pairs(width):
    idx = np.arange(width).reshape(-1,2)
    return idx[:,[1,0]].flatten()

def tie_weights(w):
    assert len(w.shape) == 3, "w is of shape (channel, num_rf_cluster, num_ori_cluster)"
    assert w.shape[1]*2 == w.shape[2], "nunm_rf_cluster * 2 = num_ori_cluster"
    width = w.shape[-1]
    ids_order = list(np.arange(width)) + list(swap_pairs(width))
    return w.repeat((1,1,2))[:,:,ids_order].reshape(-1,width,width)


class Sn_coarsen_layer(nn.Module):
    def __init__(self, cluster_ids, in_features, out_features, reflect=False):
        '''
        Assume 1-layer linear model 
        Rewrite for simple 1d case
        Generalize to d_in, d_out
        '''
        super(Sn_coarsen_layer, self).__init__()
        self.reflect = reflect
        if self.reflect:
            print("reflecting!")
            self.ori_cluster_ids = cluster_ids #original
            #merge clusters that are reflection pairs of each other
            self.cluster_ids, self.pairs = reflection_clusters(cluster_ids)
            size_check = [len(c) for c in self.cluster_ids]
            assert all(elem == size_check[0] for elem in size_check), "cannot apply this simple approach for unbalanced cluster!"
        else:
            self.cluster_ids = cluster_ids

        self.permute_ids = flatten_ids(self.cluster_ids) #rearrange nodes
        self.inv_permute = inv(self.permute_ids) #inverse permutation
        self.num_cluster = len(self.cluster_ids)
        self.num_nodes = len(self.permute_ids)
        self.ratio = len(self.cluster_ids[0]) #cluster size
        self.in_features = in_features
        self.out_features = out_features
        self.w_diag = nn.Parameter(torch.rand(self.in_features * self.out_features, self.num_cluster))
        self.w_off = nn.Parameter(torch.rand(self.in_features * self.out_features, self.num_cluster, self.num_cluster)) #TODO- rand vs ones?
              
        self.b1 = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
        stdv = 1. / math.sqrt(out_features)
        self.b1.data.uniform_(-stdv, stdv)
        #TODO: might be fun to check the learned cluster message matrix!

        #self.reset_weight()
    
    def reset_weight(self):
        stdv = 1. / math.sqrt(self.out_features) #potential big game changer! (before forget self.n)
        self.w_diag.data.uniform_(-stdv, stdv)
        self.w_off.data.uniform_(-stdv, stdv) #smaller, proportional

    def forward(self, x):
        #layer 1 updates
        bs, num_nodes, in_features = x.shape
        M_diag = torch.eye(self.ratio).to(x.device)
        M_off = torch.ones((self.ratio, self.ratio)).to(x.device)
        #first make the diagonal matrix, then broadcast by kron, then rearrange to tile
        W_diag = rearrange(torch.kron(torch.diag_embed(self.w_diag), M_diag), \
                           '(i1 i2) j k -> (i1 j) (i2 k)', i1=self.out_features) #
        
        W_off = (1/ self.num_nodes) * rearrange(torch.kron(self.w_off, M_off), \
                           '(i1 i2) j k -> (i1 j) (i2 k)', i1=self.out_features)
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


class Sn_coarsen_layer_reflect(nn.Module):
    def __init__(self, cluster_ids, in_features, out_features):
        '''
        Assume 1-layer linear model 
        Rewrite for simple 1d case
        Generalize to reflection by 
        '''
        super(Sn_coarsen_layer_reflect, self).__init__()

        self.ori_cluster_ids = cluster_ids #original
        #merge clusters that are reflection pairs of each other
        self.cluster_ids, self.pairs = reflection_clusters(cluster_ids)
        size_check = [len(c) for c in self.cluster_ids]
        assert all(elem == size_check[0] for elem in size_check), "cannot apply this simple approach for unbalanced cluster!"
        print(len(self.cluster_ids), len(self.ori_cluster_ids))
        assert len(self.cluster_ids)*2 == len(self.ori_cluster_ids), "reflection ratio not correct!"

        self.permute_ids = flatten_ids(self.cluster_ids) #rearrange nodes, with order (cluster, cluster_rf)
        self.inv_permute = inv(self.permute_ids) #inverse permutation
        self.num_cluster = len(self.cluster_ids)
        self.num_nodes = len(self.permute_ids)
        self.ratio = len(self.ori_cluster_ids[0]) #cluster size
        self.in_features = in_features
        self.out_features = out_features
        self.w_diag = nn.Parameter(torch.rand(self.in_features * self.out_features, self.num_cluster))
        self.w_off = nn.Parameter(torch.rand(self.in_features * self.out_features, self.num_cluster, 
                                              self.num_cluster*2)) 
              
        self.b1 = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
        stdv = 1. / math.sqrt(out_features)
        self.b1.data.uniform_(-stdv, stdv)
        #TODO: might be fun to check the learned cluster message matrix!

        #self.reset_weight()
    
    def reset_weight(self):
        stdv = 1. / math.sqrt(self.out_features) #potential big game changer! (before forget self.n)
        self.w_diag.data.uniform_(-stdv, stdv)
        self.w_off.data.uniform_(-stdv, stdv) #smaller, proportional

    def forward(self, x):
        #layer 1 updates
        bs, num_nodes, in_features = x.shape
        M_diag = torch.eye(self.ratio).to(x.device)
        M_off = torch.ones((self.ratio, self.ratio)).to(x.device)
        #first make the diagonal matrix, then broadcast by kron, then rearrange to tile
        W_diag = rearrange(torch.kron(torch.diag_embed(self.w_diag.repeat_interleave(repeats=2, dim=1)), M_diag), \
                           '(i1 i2) j k -> (i1 j) (i2 k)', i1=self.out_features) #
        
        W_off = (1/ self.num_nodes) * rearrange(torch.kron(tie_weights(self.w_off), M_off), \
                           '(i1 i2) j k -> (i1 j) (i2 k)', i1=self.out_features)
        
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
    def __init__(self, cluster_ids, in_features, out_features, hid_dim, reflect=False):
        '''
        If reflect:incorporate additional global reflection symmetry via weight tying
        Caveat: Not ALL equiv function for reflection sym.
         (nonetheless, form a nested sequence of hypothesis class with increasing coarsening size)
        '''
        super(Sn_coarsen_net, self).__init__()
        print("Warning: this asserts patched-clustering for regular grid images!")

        self.cluster_ids = cluster_ids
        self.permute_ids = flatten_ids(cluster_ids) #rearrange nodes
        self.inv_permute = inv(self.permute_ids) #inverse permutation
        self.num_cluster = len(cluster_ids)
        self.num_nodes = len(self.permute_ids)
        self.ratio = len(self.cluster_ids[0]) #cluster size
        self.reflect = reflect
        #layers
        if self.reflect:
            self.input_layer = Sn_coarsen_layer_reflect(cluster_ids, in_features, hid_dim)
            self.output_layer = Sn_coarsen_layer_reflect(cluster_ids, hid_dim, out_features)
        else:
            self.input_layer = Sn_coarsen_layer(cluster_ids, in_features, hid_dim)
            self.output_layer = Sn_coarsen_layer(cluster_ids, hid_dim, out_features)                        
        # self.input_layer = Sn_coarsen_layer(cluster_ids, in_features, hid_dim, reflect)
        # self.output_layer = Sn_coarsen_layer(cluster_ids, hid_dim, out_features, reflect)
        self.reset_weight()

    def reset_weight(self):
        for layer in [self.input_layer, self.output_layer]:
            layer.reset_weight()    
    
    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.output_layer(x)

        return x
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


def get_clusters(N, cut=7):
    '''
    N: width/height of the grid, 28
    cut: number of chunks per width/height
    '''
    assert N % cut == 0, "must pass in cut that divides with N"
    cuts = list(range(0,N+cut,cut))
    inds = [(i,j) for i, j in zip(cuts[:-1], cuts[1:])]
    clusters = []
    for i, j in inds:
        for m, n in inds:
            clusters.append([i, j, m, n])
    return clusters

def get_cluster_idxs(N, cut, verbose=False):
    '''
    Get it in a row-wise manner (when flattening the 28*28 grid into 764-vector)
    '''
    total_num = N*N
    mat = np.arange(total_num).reshape(N, N)
    if verbose:
        print(mat)
    side_length = int(N//cut) #N=28, cut=2, side_length=14
    ids = []
    for c_row in range(cut):
        for c_col in range(cut):
            cluster = []
            for i in range(side_length*c_row, side_length*(c_row+1)):
                for j in range(side_length*c_col, side_length*(c_col+1)):
                    cluster.append(mat[i,j])
            ids.append(cluster)
    return ids

def reflection_clusters(c_ids):
    n = len(c_ids)
    h = int(np.sqrt(n))
    w = int(np.sqrt(n))
    assert h == w, "only deal with regular grid here!"
    h_index = list(np.arange(h))
    merged_cluster = []
    pairs = []
    for i, j in zip(h_index[:h//2], h_index[::-1][:h//2]):
        for w_id in np.arange(w):
            pairs.append([i + w_id*h,j + w_id*h])
            merged_cluster.append(c_ids[i + w_id*h] + c_ids[ j + w_id*h])
    if h % 2 == 1:
        #deal with odd side, signleton cluster in the middle
        for w_id in np.arange(w):
            pairs.append([h//2 + w_id*h])
            merged_cluster.append(c_ids[h//2 + w_id*h])
    #print(list(np.arange(n)))
    #print( [p for pair in pairs for p in pair].sort())
    pairs_flat =  [p for pair in pairs for p in pair]
    pairs_flat.sort()
    assert list(np.arange(n)) == pairs_flat, "missing something!"
    return merged_cluster, pairs

def get_masked_img(img, img_size=28, mask_size=7):
    """Randomly masks image"""
    #compute masked_img as x and reuse!
    background_value = img.min().item()
    y1, x1 = np.random.randint(0, img_size - mask_size, 2)
    y2, x2 = y1 + mask_size, x1 + mask_size
    masked_part = img[:, y1:y2, x1:x2]
    masked_img = img.clone()
    masked_img[:, y1:y2, x1:x2] = 0 #background_value #TODO: check if 1 or 0
    return masked_img

def get_masked_images(images, mask_size=14):
    '''
    images: batch_size, img_shape
    '''
    masked_images = []
    img_size = images[0].shape[-1]
    for img in images:
        out = get_masked_img(img, img_size=img_size, mask_size=mask_size)
        out = out.reshape(1,-1,img_size, img_size)
        masked_images.append(out)
    masked_images = torch.cat(masked_images)
    return masked_images#.unsqueeze(1)


def generate_grid_graph(size=28):
    #src: https://stackoverflow.com/questions/71866862/problem-with-adjacency-matrix-in-2d-grid-graph-in-python
    G = nx.grid_2d_graph(size, size)
    #sorted(G,)
    #nx.draw(G, with_labels=True)
    #print(G.nodes())
    A = nx.to_numpy_array(G) 
    return A

def prepare_data(train_loader, test_loader, device, mask_size):
    train_y, _ = next(iter(train_loader))
    test_y, _ = next(iter(test_loader))
    train_x = get_masked_images(train_y, mask_size=mask_size)
    test_x = get_masked_images(test_y, mask_size=mask_size)
    #reshape and move to gpu
    train_y = train_y.flatten(start_dim=2).permute(0,2,1) # (bs, 1, 28, 28) -> (bs, 1, 28*28) -> #(bs, 28*28, 1)
    train_x = train_x.flatten(start_dim=2).permute(0,2,1) # (bs, 1, 28, 28) -> (bs, 1, 28*28) -> #(bs, 28*28, 1)
    train_x, train_y = train_x.to(device), train_y.to(device)

    test_y = test_y.flatten(start_dim=2).permute(0,2,1) # (bs, 1, 28, 28) -> (bs, 1, 28*28) -> #(bs, 28*28, 1)
    test_x = test_x.flatten(start_dim=2).permute(0,2,1) # (bs, 1, 28, 28) -> (bs, 1, 28*28) -> #(bs, 28*28, 1)
    test_x, test_y = test_x.to(device), test_y.to(device)
    
    shuffle_idx = torch.randperm(train_x.shape[0])
    train_x = train_x[shuffle_idx, :, :]
    train_y = train_y[shuffle_idx, :, :]
    #split into train/val after random shuffling
    train_size = int(len(shuffle_idx)*0.8)
    training_x = train_x[:train_size, :, :]
    val_x = train_x[train_size:, :, :]
    training_y = train_y[:train_size, :, :]
    val_y = train_y[train_size:, :, :]   

    return training_x, training_y, val_x, val_y, test_x, test_y


def run_exp(model, train_x, train_y, val_x, val_y, test_x, test_y, device, 
            run=0, n_epochs=1000,lr=0.1, mom=0.5, decay=0, mask_size=7):
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #train_losses, test_losses = [], []
    best_val, best_test = 1e5, 1e5
    for epoch in range(n_epochs):
        #begin training
        model.train()
        optimizer.zero_grad()
        output = model(train_x)
        loss = F.mse_loss(output, train_y)
        loss.backward()
        optimizer.step()
        # if (epoch + 1) % 100 == 0:
        #     print(model.w_diag.data)
        #     print(model.w_off.data)
        #begin testing        
        model.eval()
        output_val = model(val_x)
        val_loss = F.mse_loss(output_val, val_y)
        if val_loss < best_val:
            best_val = val_loss 
            output_test = model(test_x)
            best_test =  F.mse_loss(output_test, test_y) # sum up batch loss    

        if (epoch+1) % n_epochs == 0:
            print(f"best_val={best_val:.4f}, bes_test={best_test:.4f}")
        #test_losses.append(test_loss.item())
    #return train_losses, test_losses, model
    return best_val, best_test, model
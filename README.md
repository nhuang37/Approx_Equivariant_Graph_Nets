# Approximately Equivariant Graph Networks

This repo contains codes for paper "Approximately Equivariant Graph Networks" (https://arxiv.org/abs/2308.10436)

## Dependencies
- Python 3.7+
- Pytorch 1.10+

## Experiments



### 0. Simulations
- Fig 1.: Symmetry model selection example for $f: \mathbb R^3 \to \mathbb R$, where $f$ can be non-invariant, $S_2$-invaraint, or $S_3$-invariant. See ```project_subgroup.ipynb```

### 1. Image Inpainting
See folder ```./Image_inpainting```
- Figure 6 (left): Bias-Variance tradeoff via graph coarsening
  - To reproduce the results, see notebook ```graph_coarsen_runs.ipynb```
  - The linear model baselines are implemented in notebook ```graph_coarsen_runs_linear.ipynb```
- Figure 6 (right): Ablation of coarsened graph symmetry
  - To reproduce the results, see notebook ```FASHION+hflip_inpainting_reflection.ipynb```

### 2. Traffic Forecasting: 
See folder ```./DCRNN_Pytorch```
The model architecture and data are adapated from: https://github.com/chnsh/DCRNN_PyTorch

(a) Download the following data folders and store them to ```./DCRNN_Pytorch/data```
- METR-LA-T3 : G drive link [here](https://drive.google.com/drive/folders/1TiGfCf_CTr2WZ0lK0C9XUDLU-GjprBRo?usp=share_link) to download and store the traffic graph signals, with using (T-3, T-2, T-1) graph signals to predict (T, T+1, T+2) graph signals. Data credit to SOURCE
- sensor_graph : G drive link [here](https://drive.google.com/drive/folders/139d3quRQkC08zoxVID7AIWPcfr74_KK7?usp=sharing) to download and store the graph adjacency files

(b) Run **aut(G)**-equivariant DCRNN with the default set-up (c.f. Table 3): 
```
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la_aut_gc_false_t3.yaml --aut True
```
  - To use the sparsified graph adjacency, change the above to ```config_filename=data/model/dcrnn_la_sp_aut_gc_false_t3.yaml```
  - To compare with the vanilla DCRNN, remove ```--aut True```

#### File directory:

  ```./DCRNN_PyTorch```
  - ```orbit_idx.p``` : store the 2-cluster assignments 
  - ```orbit_idx_9.p```: store the 9-cluster assignment
  - ```dcrnn_train_pytorch.py``` : main file to run the experiment

  ```./DCRNN_Pytorch/data/model```
  - ```dcrnn_la_aut_gc_false_t3.yaml``` : config file to run experiments on the original traffic graph
  - ```dcrnn_la_sp_aut_gc_false_t3.yaml``` : config file to run experiments on the sparsified traffic graph

  ```./DCRNN_Pytorch/model/pytorch```
  - ```dcrnn_supervisor.py``` : training script
  - ```dcrnn_model.py``` : DCRNN spatial-temporal GNN script
  - ```dcrnn_cell.py``` : modified basic graph convolution block to allow (approximate) **aut(G) equivariance**

  
### 3. Human Pose Estimation
See folder ```./Human_Pose_Est```
The model architecture and data are adapted from: https://github.com/garyzhao/SemGCN

(a) Download data following instructions [here](https://github.com/garyzhao/SemGCN/blob/master/data/README.md)

(b) Run models with different symmetry choices, using $\mathcal{G}$-Net(gc+ew) with graph convolution and learnable edge weights (c.f. Table 4):
  - $\mathcal{S}_{16}$ (Default SemGCN setup): ```python3 main_gcn_aut.py --epochs 30 --hid_dim 128  --checkpoint "./checkpoints"```
  - Relax-$\mathcal{S}_{16}$: ```python3 main_gcn_aut.py --no_tie --epochs 30 --hid_dim 128  --checkpoint "./checkpoints"```
  - $\mathcal{S}_2^2$ (aut(G)): ```python3 main_gcn_aut.py --aut --epochs 30 --hid_dim 128  --checkpoint "./checkpoints"```
  - Trivial: ```python3 main_gcn_aut.py --triv --epochs 30 --hid_dim 128  --checkpoint "./checkpoints"```

(c) Run different model variants (c.f. Table 5):
  - Vanilla $\mathcal{G}$-Net: ```python3 main_gcn_aut.py --no_gc --no_ew  --epochs 30 --hid_dim 128  --checkpoint "./checkpoints"```
  - $\mathcal{G}$-Net(gc): ```python3 main_gcn_aut.py --no_ew  --epochs 30 --hid_dim 128  --checkpoint "./checkpoints"```
  - $\mathcal{G}$-Net(pt): ```python3 main_gcn_aut.py --no_gc  --no_ew --pointwise --epochs 30 --hid_dim 128  --checkpoint "./checkpoints"```

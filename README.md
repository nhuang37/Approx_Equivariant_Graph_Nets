# Bias-Variance Tradeoff in Learning Approximately Equivariant Graph Networks

This repo contains codes for paper "On the Bias-Variance Tradeoff in Learning Approximately Equivariant Graph Networks"

## Dependencies
- Python 3.7+
- Pytorch 1.10+
- [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

## Experiments

### 1. Human Pose Estimation

### 2. Traffic Forecasting: See folder ```./DCRNN_Pytorch```
(a) Download the following data folders and store them to ```./DCRNN_Pytorch/data```
- METR-LA-T3 : G drive link [here](https://drive.google.com/drive/folders/1TiGfCf_CTr2WZ0lK0C9XUDLU-GjprBRo?usp=share_link) to download and store the traffic graph signals, with using (T-3, T-2, T-1) graph signals to predict (T, T+1, T+2) graph signals. Data credit to SOURCE
- sensor_graph : G drive link [here](https://drive.google.com/drive/folders/139d3quRQkC08zoxVID7AIWPcfr74_KK7?usp=sharing) to download and store the graph adjacency files

(b) Run **aut(G)**-equivariant DCRNN with the default set-up: 
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


### 3. Simulations
- Fig 1.: Symmetry model selection example for $f: \mathbb R^3 \to \mathbb R$, where $f$ can be non-invariant, $S_2$-invaraint, or $S_3$-invariant. See ```project_subgroup.ipynb```

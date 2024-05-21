# Unlink to Unlearn: Simplifying Edge Unlearning in GNNs
Code repository of "[Unlink to Unlearn: Simplifying Edge Unlearning in GNNs](https://dl.acm.org/doi/10.1145/3589335.3651578)".

## Environment
```
torch == 2.1.0
torch_geometric == 2.4.0
numpy == 1.26.2
sklearn == 1.3.2
networkx == 3.0
ogb ==1.3.6
wandb
tqdm
```

## Usage

More training arguments can be seen at `framework/training_args.py`. 

### Training

```python
python train_gnn.py --dataset Cora
```

### Unlearning

```python
python delete_gnn.py --dataset Cora --df_size 5.0 --unlearning_model utu 
```

## Thanks
Some of the code was forked from the code repository of [GNNDelete](https://github.com/mims-harvard/GNNDelete/). 

## Citation
```
@inproceedings{tan2024unlink,
author = {Tan, Jiajun and Sun, Fei and Qiu, Ruichen and Su, Du and Shen, Huawei},
title = {Unlink to Unlearn: Simplifying Edge Unlearning in GNNs},
year = {2024},
doi = {10.1145/3543873.3587375},
booktitle = {Companion Proceedings of the ACM on Web Conference 2024},
pages = {489–492},
series = {WWW '24},
}
```

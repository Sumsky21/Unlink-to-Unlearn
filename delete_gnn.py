import os
import copy
import json
import wandb
import pickle
import argparse
import torch
import torch.nn as nn
from torch_geometric.utils import to_undirected, to_networkx, k_hop_subgraph, is_undirected
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.seed import seed_everything

from framework import get_model, get_trainer
from framework.training_args import parse_args
from framework.utils import *
from framework.data_loader import split_forget_retain, train_test_split_edges_no_neg_adj_mask, get_original_data
# from train_mi import load_mi_models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_processed_data(d, val_ratio, test_ratio, df_ratio, subset='in'):
    '''pend for future use'''
    data = get_original_data(d)

    data = train_test_split_edges_no_neg_adj_mask(data, val_ratio, test_ratio)
    data = split_forget_retain(data, df_ratio, subset)
    return data


torch.autograd.set_detect_anomaly(True)
def main():
    args = parse_args()
    original_path = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'original', str(args.random_seed))
    attack_path_all = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'member_infer_all', str(args.random_seed))
    args.attack_dir = attack_path_all
    if not os.path.exists(attack_path_all):
        os.makedirs(attack_path_all, exist_ok=True)
    shadow_path_all = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'shadow_all', str(args.random_seed))
    args.shadow_dir = shadow_path_all
    if not os.path.exists(shadow_path_all):
        os.makedirs(shadow_path_all, exist_ok=True)

    args.checkpoint_dir = os.path.join(
        args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model, 
        '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    seed_everything(args.random_seed)

    # Dataset
    data = get_processed_data(args.dataset, val_ratio=0.05, test_ratio=0.05, df_ratio=args.df_size)
    print('Directed dataset:', data)
    if args.gnn not in ['rgcn', 'rgat']:
        args.in_dim = data.x.shape[1]

    print('Training args', args)
    

    # Model
    model = get_model(args, data.sdf_node_1hop_mask, data.sdf_node_2hop_mask, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)

    if args.unlearning_model != 'retrain':  # Start from trained GNN model
        if os.path.exists(os.path.join(original_path, 'pred_proba.pt')):
            logits_ori = torch.load(os.path.join(original_path, 'pred_proba.pt'))   # logits_ori: tensor.shape([num_nodes, num_nodes]), represent probability of edge existence between any two nodes
            if logits_ori is not None:
                logits_ori = logits_ori.to(device)
        else:
            logits_ori = None

        model_ckpt = torch.load(os.path.join(original_path, 'model_best.pt'), map_location=device)
        model.load_state_dict(model_ckpt['model_state'], strict=False)
   
    else:       # Initialize a new GNN model
        retrain = None
        logits_ori = None

    model = model.to(device)
    # data = data.to(device)

    # Optimizer
    if 'gnndelete' in args.unlearning_model:
        parameters_to_optimize = [
            {'params': [p for n, p in model.named_parameters() if 'del' in n], 'weight_decay': args.weight_decay}
        ]
        print('parameters_to_optimize', [n for n, p in model.named_parameters() if 'del' in n])
        if 'layerwise' in args.loss_type:
            optimizer1 = torch.optim.Adam(model.deletion1.parameters(), lr=args.lr)
            optimizer2 = torch.optim.Adam(model.deletion2.parameters(), lr=args.lr)
            optimizer = [optimizer1, optimizer2]
        else:
            optimizer = torch.optim.Adam(parameters_to_optimize, lr=args.lr)
    else:
        parameters_to_optimize = [
            {'params': [p for n, p in model.named_parameters()], 'weight_decay': args.weight_decay}
        ]
        print('parameters_to_optimize', [n for n, p in model.named_parameters()])
        optimizer = torch.optim.Adam(parameters_to_optimize, lr=args.lr)
    
    wandb.init(config=args, project="GNNDelete", group="over_unlearn", name=get_run_name(args), mode=args.mode)
    wandb.watch(model, log_freq=100)


    # MI attack model
    attack_model_all = None
    attack_model_sub = None

    # Train
    trainer = get_trainer(args)
    trainer.train(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

    # Test
    if args.unlearning_model != 'retrain':
        retrain_path = os.path.join(
            'checkpoint', args.dataset, args.gnn, 'retrain', 
            '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]), 
            'model_best.pt')
        if os.path.exists(retrain_path):
            retrain_ckpt = torch.load(retrain_path, map_location=device)
            retrain_args = copy.deepcopy(args)
            retrain_args.unlearning_model = 'retrain'
            retrain = get_model(retrain_args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)
            retrain.load_state_dict(retrain_ckpt['model_state'])
            retrain = retrain.to(device)
            retrain.eval()
        else:
            retrain = None
    else:
        retrain = None
    
    test_results = trainer.test(model, data, model_retrain=retrain, attack_model_all=attack_model_all, attack_model_sub=attack_model_sub)
    print(test_results[-1])

    trainer.save_log()
    wandb.finish()


if __name__ == "__main__":
    main()

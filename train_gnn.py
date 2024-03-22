import os
import wandb
import pickle
import torch
from torch_geometric.seed import seed_everything
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.datasets import RelLinkPredDataset, WordNet18
from torch_geometric.seed import seed_everything

from framework import get_model, get_trainer
from framework.training_args import parse_args
from framework.trainer.base import Trainer
from framework.utils import negative_sampling_kg
from framework.data_loader import get_original_data, train_test_split_edges_no_neg_adj_mask
import networkx as nx


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_processed_data(d, val_ratio=0.05, test_ratio=0.05):
    data = get_original_data(d)
    data = train_test_split_edges_no_neg_adj_mask(data, val_ratio, test_ratio)
    return data


def main():
    args = parse_args()
    args.unlearning_model = 'original'
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model, str(args.random_seed))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    seed_everything(args.random_seed)

    # Dataset
    data = get_processed_data(args.dataset)
    if args.gnn not in ['rgcn', 'rgat']:
        args.in_dim = data.x.shape[1]


    # Use proper training data for original and Dr
    if args.gnn in ['rgcn', 'rgat']:
        if not hasattr(data, 'train_mask'):
            data.train_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)

        # data.dtrain_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
        # data.edge_index_mask = data.dtrain_mask.repeat(2)
        
    else:
        data.dtrain_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)

    # To undirected
    train_pos_edge_index = to_undirected(data.train_pos_edge_index)
    data.train_pos_edge_index = train_pos_edge_index
    data.dtrain_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    assert is_undirected(data.train_pos_edge_index)


    print('Undirected dataset:', data)
    wandb.init(config=args, group="learning", name="{}_{}".format(args.dataset, args.gnn), mode=args.mode)
    # Model
    model = get_model(args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type).to(device)
    wandb.watch(model, log_freq=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=args.weight_decay)

    # Train
    trainer = get_trainer(args)
    trainer.train(model, data, optimizer, args)

    # Test
    test_results = trainer.test(model, data)
    trainer.save_log()
    print(test_results[-1])
    wandb.finish()


if __name__ == "__main__":
    main()

import numpy as np
import torch
import networkx as nx
import wandb
import os


def wandb_log(log_dict):
    wandb.log({key.replace("_", "/", 1): value for key, value in log_dict.items()})

def get_node_edge(graph):
    degree_sorted_ascend = sorted(graph.degree, key=lambda x: x[1])

    return degree_sorted_ascend[-1][0]

def h_hop_neighbor(G, node, h):
    path_lengths = nx.single_source_dijkstra_path_length(G, node)
    return [node for node, length in path_lengths.items() if length == h]
                    
def get_enclosing_subgraph(graph, edge_to_delete):
    subgraph = {0: [edge_to_delete]}
    s, t = edge_to_delete
    
    neighbor_s = []
    neighbor_t = []
    for h in range(1, 2+1):
        neighbor_s += h_hop_neighbor(graph, s, h)
        neighbor_t += h_hop_neighbor(graph, t, h)
        
        nodes = neighbor_s + neighbor_t + [s, t]
        
        subgraph[h] = list(graph.subgraph(nodes).edges())
        
    return subgraph

@torch.no_grad()
def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=pos_edge_index.device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

@torch.no_grad()
def get_link_labels_kg(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=pos_edge_index.device)
    link_labels[:pos_edge_index.size(1)] = 1.

    return link_labels

@torch.no_grad()
def negative_sampling_kg(edge_index, edge_type):
    '''Generate negative samples but keep the node type the same'''

    edge_index_copy = edge_index.clone()
    for et in edge_type.unique():
        mask = (edge_type == et)
        old_source = edge_index_copy[0, mask]
        new_index = torch.randperm(old_source.shape[0])
        new_source = old_source[new_index]
        edge_index_copy[0, mask] = new_source
    
    return edge_index_copy

def get_run_name(args):
    model_unlearn_pair = args.gnn + '/' + args.unlearning_model
    name = '_'.join([args.dataset, model_unlearn_pair, args.df+str(args.df_size)])
    if args.suffix != None:
        name += '_{}'.format(args.suffix)
    return name


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience. 
    From https://github.com/Bjarten/early-stopping-pytorch.
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, score, model, z=None):

        # score = -score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, z)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, z)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, z):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        ckpt = {
            'model_state': model.state_dict(),
            # 'optimizer_state': [optimizer[0].state_dict(), optimizer[1].state_dict()],
        }
        torch.save(z, os.path.join(self.path, 'node_embeddings.pt'))
        torch.save(ckpt, os.path.join(self.path, 'model_best.pt'))
        self.val_loss_min = val_loss

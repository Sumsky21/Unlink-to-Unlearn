from ogb.linkproppred import PygLinkPropPredDataset # must put this in 1st line to avoid stall on running
import os
import math
import torch
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.utils import k_hop_subgraph, is_undirected, to_undirected, negative_sampling, subgraph
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import CitationFull, Coauthor, Amazon, Planetoid, Reddit2, Flickr

def get_original_data(d):
    data_dir = './data'    # 改为自定的目录

    if d in ['Cora', 'PubMed', 'DBLP']:
        dataset = CitationFull(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
    elif d in ['Cora_p', 'PubMed_p', 'Citeseer_p']:
        dataset = Planetoid(os.path.join(data_dir, d), d.split('_')[0], transform=T.NormalizeFeatures())
    elif d in ['CS', 'Physics']:
        dataset = Coauthor(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
    elif d in ['Amazon']:
        dataset = Amazon(os.path.join(data_dir, d), 'Photo', transform=T.NormalizeFeatures())
    elif d in ['Reddit']:   # On 4090: need minibatch
        dataset = Reddit2(os.path.join(data_dir, d), transform=T.NormalizeFeatures())
    elif d in ['Flickr']:
        dataset = Flickr(os.path.join(data_dir, d), transform=T.NormalizeFeatures())
    elif 'ogbl' in d:
        dataset = PygLinkPropPredDataset(root=os.path.join(data_dir, d), name=d)
    else:
        raise NotImplementedError(f"{d} not supported.")
    data = dataset[0]
    return data


def gen_inout_mask(data):
    _, local_edges, _, mask = k_hop_subgraph(
        data.val_pos_edge_index.flatten().unique(), 
        2, 
        data.train_pos_edge_index, 
        num_nodes=data.num_nodes)
    distant_edges = data.train_pos_edge_index[:, ~mask]
    print('Number of edges. Local: ', local_edges.shape[1], 'Distant:', distant_edges.shape[1])

    in_mask = mask
    out_mask = ~mask

    return {'in': in_mask, 'out': out_mask}


def split_forget_retain(data, df_size, subset='in'):
    if df_size >= 100:     # df_size is number of nodes/edges to be deleted
        df_size = int(df_size)
    else:                       # df_size is the ratio
        df_size = int(df_size / 100 * data.train_pos_edge_index.shape[1])
    print(f'Original size: {data.train_pos_edge_index.shape[1]:,}')
    print(f'Df size: {df_size:,}')
    df_mask_all = gen_inout_mask(data)[subset]
    df_nonzero = df_mask_all.nonzero().squeeze()        # subgraph子图内/外的edge idx序号

    idx = torch.randperm(df_nonzero.shape[0])[:df_size]
    df_global_idx = df_nonzero[idx]

    dr_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    dr_mask[df_global_idx] = False

    df_mask = torch.zeros(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    df_mask[df_global_idx] = True

    # Collect enclosing subgraph of Df for loss computation
    # Edges in S_Df
    _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(), 
        2, 
        data.train_pos_edge_index,
        num_nodes=data.num_nodes)
    data.sdf_mask = two_hop_mask

    # Nodes in S_Df
    _, one_hop_edge, _, one_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(), 
        1, 
        data.train_pos_edge_index,
        num_nodes=data.num_nodes)
    sdf_node_1hop = torch.zeros(data.num_nodes, dtype=torch.bool)
    sdf_node_2hop = torch.zeros(data.num_nodes, dtype=torch.bool)

    sdf_node_1hop[one_hop_edge.flatten().unique()] = True
    sdf_node_2hop[two_hop_edge.flatten().unique()] = True

    assert sdf_node_1hop.sum() == len(one_hop_edge.flatten().unique())
    assert sdf_node_2hop.sum() == len(two_hop_edge.flatten().unique())

    data.sdf_node_1hop_mask = sdf_node_1hop
    data.sdf_node_2hop_mask = sdf_node_2hop

    assert not is_undirected(data.train_pos_edge_index)

    train_pos_edge_index, [df_mask, two_hop_mask] = to_undirected(data.train_pos_edge_index, [df_mask.int(), two_hop_mask.int()])
    # to_undirected return full undirected edges and corresponding mask for given edge_attrs
    two_hop_mask = two_hop_mask.bool()  
    df_mask = df_mask.bool()
    dr_mask = ~df_mask

    data.train_pos_edge_index = train_pos_edge_index
    data.edge_index = train_pos_edge_index
    assert is_undirected(data.train_pos_edge_index)

    data.directed_df_edge_index = data.train_pos_edge_index[:, df_mask]
    data.sdf_mask = two_hop_mask
    data.df_mask = df_mask
    data.dr_mask = dr_mask
    return data


def split_shadow_target(data):
    # 检查是否存在标签y
    if data.y != None:
        # 获取节点的标签
        labels = data.y.numpy()

        # 创建StratifiedShuffleSplit实例，以确保每个类别按比例分割
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

        # 获取分割后的节点索引
        for target_index, shadow_index in sss.split(torch.zeros(labels.shape[0]), labels):
            target_nodes = torch.from_numpy(target_index)
            shadow_nodes = torch.from_numpy(shadow_index)
    else:
        # 如果没有标签y，则使用ShuffleSplit进行随机划分
        ss = ShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

        # 获取所有节点的索引
        all_indices = torch.arange(data.x.shape[0])

        # 获取分割后的节点索引
        for target_index, shadow_index in ss.split(all_indices):
            target_nodes = all_indices[target_index]
            shadow_nodes = all_indices[shadow_index]

    # 使用torch_geometric.utils.subgraph创建子图
    target_edge_index, target_edge_attr = subgraph(
        target_nodes, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True
    )
    shadow_edge_index, shadow_edge_attr = subgraph(
        shadow_nodes, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True
    )

    # 创建子图数据对象
    target_data = Data(
        x=data.x[target_nodes],
        edge_index=target_edge_index,
        edge_attr=target_edge_attr,
        y=data.y[target_nodes] if data.y != None else None,
    )
    shadow_data = Data(
        x=data.x[shadow_nodes],
        edge_index=shadow_edge_index,
        edge_attr=shadow_edge_attr,
        y=data.y[shadow_nodes] if data.y != None else None,
    )
    # 现在target_data和shadow_data分别代表原图的两个子图
    return shadow_data, target_data


def train_test_split_edges_no_neg_adj_mask(data, val_ratio: float = 0.05, test_ratio: float = 0.05, two_hop_degree=None):
    '''Avoid adding neg_adj_mask'''

    num_nodes = data.num_nodes
    row, col = data.edge_index
    edge_attr = data.edge_attr

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    if two_hop_degree is not None:          # Use low degree edges for test sets
        low_degree_mask = two_hop_degree < 50

        low = low_degree_mask.nonzero().squeeze()
        high = (~low_degree_mask).nonzero().squeeze()

        low = low[torch.randperm(low.size(0))]
        high = high[torch.randperm(high.size(0))]

        perm = torch.cat([low, high])

    else:
        perm = torch.randperm(row.size(0))

    row = row[perm]
    col = col[perm]

    # Train
    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.train_pos_edge_index, data.train_pos_edge_attr = None
    else:
        data.train_pos_edge_index = data.train_pos_edge_index
    
    assert not is_undirected(data.train_pos_edge_index)

    
    # Test
    r, c = row[:n_t], col[:n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    neg_edge_index = negative_sampling(
        edge_index=data.test_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.test_pos_edge_index.shape[1])

    data.test_neg_edge_index = neg_edge_index

    # Valid
    r, c = row[n_t:n_t+n_v], col[n_t:n_t+n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)

    neg_edge_index = negative_sampling(
        edge_index=data.val_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.val_pos_edge_index.shape[1])

    data.val_neg_edge_index = neg_edge_index

    return data




def load_dict(filename):
    '''Load entity and relation to id mapping'''

    mapping = {}
    with open(filename, 'r') as f:
        for l in f:
            l = l.strip().split('\t')
            mapping[l[0]] = l[1]

    return mapping

def load_edges(filename):
    with open(filename, 'r') as f:
        r = f.readlines()
    r = [i.strip().split('\t') for i in r]

    return r

def generate_true_dict(all_triples):
    heads = {(r, t) : [] for _, r, t in all_triples}
    tails = {(h, r) : [] for h, r, _ in all_triples}

    for h, r, t in all_triples:
        heads[r, t].append(h)
        tails[h, r].append(t)

    return heads, tails

def get_loader(args, delete=[]):
    prefix = os.path.join('./data', args.dataset)

    # Edges
    train = load_edges(os.path.join(prefix, 'train.txt'))
    valid = load_edges(os.path.join(prefix, 'valid.txt'))
    test = load_edges(os.path.join(prefix, 'test.txt'))
    train = [(int(i[0]), int(i[1]), int(i[2])) for i in train]
    valid = [(int(i[0]), int(i[1]), int(i[2])) for i in valid]
    test = [(int(i[0]), int(i[1]), int(i[2])) for i in test]
    train_rev = [(int(i[2]), int(i[1]), int(i[0])) for i in train]
    valid_rev = [(int(i[2]), int(i[1]), int(i[0])) for i in valid]
    test_rev = [(int(i[2]), int(i[1]), int(i[0])) for i in test]
    train = train + train_rev
    valid = valid + valid_rev
    test = test + test_rev
    all_edge = train + valid + test

    true_triples = generate_true_dict(all_edge)

    edge = torch.tensor([(int(i[0]), int(i[2])) for i in all_edge], dtype=torch.long).t()
    edge_type = torch.tensor([int(i[1]) for i in all_edge], dtype=torch.long)#.view(-1, 1)

    # Masks
    train_size = len(train)
    valid_size = len(valid)
    test_size = len(test)
    total_size = train_size + valid_size + test_size

    train_mask = torch.zeros((total_size,)).bool()
    train_mask[:train_size] = True

    valid_mask = torch.zeros((total_size,)).bool()
    valid_mask[train_size:train_size + valid_size] = True
    
    test_mask = torch.zeros((total_size,)).bool()
    test_mask[-test_size:] = True

    # Graph size
    num_nodes = edge.flatten().unique().shape[0]
    num_edges = edge.shape[1]
    num_edge_type = edge_type.unique().shape[0]

    # Node feature
    x = torch.rand((num_nodes, args.in_dim))

    # Delete edges
    if len(delete) > 0:
        delete_idx = torch.tensor(delete, dtype=torch.long)
        num_train_edges = train_size // 2
        train_mask[delete_idx] = False
        train_mask[delete_idx + num_train_edges] = False
        train_size -= 2 * len(delete)
    
    node_id = torch.arange(num_nodes)
    dataset = Data(
        edge_index=edge, edge_type=edge_type, x=x, node_id=node_id, 
        train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)

    dataloader = GraphSAINTRandomWalkSampler(
        dataset, batch_size=args.batch_size, walk_length=args.walk_length, num_steps=args.num_steps)

    print(f'Dataset: {args.dataset}, Num nodes: {num_nodes}, Num edges: {num_edges//2}, Num relation types: {num_edge_type}')
    print(f'Train edges: {train_size//2}, Valid edges: {valid_size//2}, Test edges: {test_size//2}')
    
    return dataloader, valid, test, true_triples, num_nodes, num_edges, num_edge_type



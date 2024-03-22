from .models import GCN, GAT, GIN, GCNDelete, GATDelete, GINDelete
from .trainer.base import Trainer, KGTrainer, NodeClassificationTrainer
from .trainer.retrain import RetrainTrainer, KGRetrainTrainer
from .trainer.gnndelete_nodeemb import GNNDeleteNodeembTrainer
from .trainer.gnndelete_ni import GNNDeleteNITrainer
from .trainer.gradient_ascent import GradientAscentTrainer
from .trainer.utu import UtUTrainer
from .trainer.member_infer import MIAttackTrainer# , MIAttackTrainerNode
from .trainer.gif import GIFTrainer


trainer_mapping = {
    'original': Trainer,
    'original_node': NodeClassificationTrainer,
    'retrain': RetrainTrainer,
    'gradient_ascent': GradientAscentTrainer,
    'gnndelete_all': GNNDeleteNodeembTrainer,
    'gnndelete_ni': GNNDeleteNITrainer,
    'member_infer_all': MIAttackTrainer,
    'member_infer_sub': MIAttackTrainer,
    'mi_shadow': Trainer,
    'finetune': RetrainTrainer,
    'utu': UtUTrainer,
    'gif': GIFTrainer,
}


def get_model(args, mask_1hop=None, mask_2hop=None, num_nodes=None, num_edge_type=None):

    if 'gnndelete' in args.unlearning_model:
        model_mapping = {'gcn': GCNDelete, 'gat': GATDelete, 'gin': GINDelete}

    else:
        model_mapping = {'gcn': GCN, 'gat': GAT, 'gin': GIN}

    return model_mapping[args.gnn](args, mask_1hop=mask_1hop, mask_2hop=mask_2hop, num_nodes=num_nodes, num_edge_type=num_edge_type)


def get_trainer(args):
    return trainer_mapping[args.unlearning_model](args)
    

def get_shadow_model(args, mask_1hop=None, mask_2hop=None, num_nodes=None, num_edge_type=None):
    model_mapping = {'gcn': GCN, 'gat': GAT, 'gin': GIN}

    return model_mapping[args.gnn](args, mask_1hop=mask_1hop, mask_2hop=mask_2hop, num_nodes=num_nodes, num_edge_type=num_edge_type)


def get_attacker(args):
    return trainer_mapping['member_infer_all'](args)

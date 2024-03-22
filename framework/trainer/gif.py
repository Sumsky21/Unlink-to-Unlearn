import os
import wandb
import time
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import grad

from .base import Trainer
from ..evaluation import *
from ..utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = np.float16

class GIFTrainer(Trainer):
    '''This code is adapted from https://github.com/zleizzo/datadeletion'''

    def get_grad(self, data, model):
        # unlearn_info[2]是因删除边受到影响的节点集合，类似于sdf_node_2hop_mask这样的概念; 在边这里直接sdf_node
        print("Computing grads...")
        neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index[:, data.dr_mask],
            num_nodes=data.num_nodes,
            num_neg_samples=data.sdf_mask.sum())

        grad_all, grad1, grad2 = None, None, None
        out1 = model(data.x, data.train_pos_edge_index)
        out2 = model(data.x, data.train_pos_edge_index[:, data.dr_mask])
            
        mask1 = mask2 = data.sdf_node_2hop_mask

        logits = model.decode(out1, data.train_pos_edge_index[:, data.dr_mask], neg_edge_index)
        label = get_link_labels(data.train_pos_edge_index[:, data.dr_mask], neg_edge_index)
        loss = F.binary_cross_entropy_with_logits(logits, label, reduction='mean')
        # loss = F.nll_loss(out1, data.y, reduction='sum')

        logits = model.decode(out1, data.train_pos_edge_index[:, data.sdf_mask], neg_edge_index)
        label = get_link_labels(data.train_pos_edge_index[:, data.sdf_mask], neg_edge_index)
        loss1 = F.binary_cross_entropy_with_logits(logits, label, reduction='sum')
        # loss1 = F.nll_loss(out1[mask1], data.y[mask1], reduction='sum')

        logits = model.decode(out2, data.train_pos_edge_index[:, data.sdf_mask], neg_edge_index)
        label = get_link_labels(data.train_pos_edge_index[:, data.sdf_mask], neg_edge_index)
        loss2 = F.binary_cross_entropy_with_logits(logits, label, reduction='sum')
        # loss2 = F.nll_loss(out2[mask2], data.y[mask2], reduction='sum')

        model_params = [p for p in model.parameters() if p.requires_grad]
        grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
        grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
        grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

        return (grad_all, grad1, grad2)

    def gif_approxi(self, args, model, res_tuple):
        '''
        res_tuple == (grad_all, grad1, grad2)
        '''
        print("Unlearning model...")
        start_time = time.time()
        iteration, damp, scale = args.iteration, args.damp, args.scale

        v = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
        h_estimate = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
        for _ in range(iteration):

            model_params  = [p for p in model.parameters() if p.requires_grad]
            hv            = self.hvps(res_tuple[0], model_params, h_estimate)
            with torch.no_grad():
                h_estimate    = [ v1 + (1-damp)*h_estimate1 - hv1/scale
                            for v1, h_estimate1, hv1 in zip(v, h_estimate, hv)]

        
        # breakpoint()
        params_change = [h_est / scale for h_est in h_estimate]
        params_esti   = [p1 + p2 for p1, p2 in zip(params_change, model_params)]

        with torch.no_grad():
            for param, update in zip(model.parameters(), params_change):
                param.add_(update)  # In-place update of the model parameters

        return time.time() - start_time, model

    def hvps(self, grad_all, model_params, h_estimate):
        element_product = 0
        for grad_elem, v_elem in zip(grad_all, h_estimate):
            element_product += torch.sum(grad_elem * v_elem)
        
        return_grads = grad(element_product, model_params, create_graph=True)
        return return_grads   

    # @torch.no_grad()
    def train(self, model, data, optimizer, args, logits_ori=None, attack_model=None, attack_model_sub=None):
        # model.train()
        model, data = model.to(device), data.to(device)
        args.eval_on_cpu = False
        # MI Attack before unlearning
        if attack_model is not None:
            mi_logit_all_before, mi_sucrate_all_before = member_infer_attack(model, attack_model, data)
            self.trainer_log['mi_logit_all_before'] = mi_logit_all_before
            self.trainer_log['mi_sucrate_all_before'] = mi_sucrate_all_before
        if attack_model_sub is not None:
            mi_logit_sub_before, mi_sucrate_sub_before = member_infer_attack(model, attack_model_sub, data)
            self.trainer_log['mi_logit_sub_before'] = mi_logit_sub_before
            self.trainer_log['mi_sucrate_sub_before'] = mi_sucrate_sub_before


        grad_tuple = self.get_grad(data, model)

        time, model = self.gif_approxi(args, model, grad_tuple)
        z = model(data.x, data.train_pos_edge_index[:, data.dr_mask])

        valid_loss, auc, aup, df_auc, df_aup, df_logit, _, log = self.eval(model, data, 'val')

        log['training_time'] = time
        wandb_log(log)
        msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
        tqdm.write(' | '.join(msg))
        self.trainer_log['log'].append(log)

        # Save
        ckpt = {
            'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
            'optimizer_state': None,
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
        torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))

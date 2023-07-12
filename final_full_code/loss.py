import math
import torch
from torch import nn
import networkx as nx
import torch.nn.functional as F

class Compute_loss(nn.Module):
    def __init__(self, args, n, graph,):
        super(Compute_loss, self).__init__()
        self.labmda = args.labmda
        self.n = n
        self.device = args.device        
        self.edge_num = graph.num_edges()
        self.top_k = args.top_k
        self.top_l = args.top_l

    def contrast_loss(self, similarity, epoch=None):    
        pos_I_dis = torch.diagonal(similarity).unsqueeze(dim=0)
        pos_dis,_ = (similarity).to_sparse().values().unsqueeze(dim=0).topk(k=self.top_k, dim=1, largest=True, sorted=True)
        pos_dis = torch.cat([pos_I_dis, pos_dis], dim=1)
        neg_dis,_ = (similarity).to_sparse().values().unsqueeze(dim=0).topk(k=self.top_l, dim=1, largest=False, sorted=True)

        logits = torch.cat([pos_dis, neg_dis], dim=1) 
        lbl_1 = torch.ones(1, pos_dis.shape[1]).to(self.device)
        lbl_0 = torch.zeros(1, neg_dis.shape[1]).to(self.device)

        targets = torch.cat((lbl_1, lbl_0), 1)

        loss = nn.BCEWithLogitsLoss()(logits, targets)
        return loss

    def forward(self, epoch, z1, z2):
        similarity = torch.mm(z1, z2.T)
        loss = self.contrast_loss(similarity, epoch)
        loss += self.labmda*nn.MSELoss()(z1, z2)*self.n
        return loss
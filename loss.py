import math
import torch
from torch import nn
import networkx as nx
import torch.nn.functional as F


class Compute_loss(nn.Module):
    def __init__(self, args, n, graph,):
        super(Compute_loss, self).__init__()
        self.beta = args.beta
        self.wk = args.wk
        self.wb = args.wb
        self.n = n
        self.device = args.device
        self._eye = torch.eye(self.n, device=self.device) 
        self._ones = torch.ones(self.n, self.n, device=self.device) 
        
        self.edge_num = graph.num_edges()
        self.adj = graph.adjacency_matrix(transpose=True).to_dense().to(self.device)
        self.adj_I = self.adj + self._eye  
        self.unadj = self._ones - self.adj_I
        self.neg_n = min(args.neg_n, (self.n*(self.n-1) - self.edge_num))

    def contrast_loss(self, similarity, epoch=None):    
        self.pos_n = min(math.ceil(self.wk*epoch+self.wb), self.edge_num)
        pos_I_dis = torch.diagonal(similarity).unsqueeze(dim=0)
        pos_dis,_ = (similarity).to_sparse().values().unsqueeze(dim=0).topk(k=self.pos_n, dim=1, largest=True, sorted=True)
        pos_dis = torch.cat([pos_I_dis, pos_dis], dim=1)
        neg_dis,_ = (similarity).to_sparse().values().unsqueeze(dim=0).topk(k=self.neg_n, dim=1, largest=False, sorted=True)

        logits = torch.cat([pos_dis, neg_dis], dim=1) 
        lbl_1 = torch.ones(1, pos_dis.shape[1]).to(self.device)
        lbl_0 = torch.zeros(1, neg_dis.shape[1]).to(self.device)

        targets = torch.cat((lbl_1, lbl_0), 1)

        loss = nn.BCEWithLogitsLoss()(logits, targets) 
        # loss = nn.BCEWithLogitsLoss()(similarity, self._eye) # for Pubmed

        return loss

    def forward(self, epoch, z1, z2):
        similarity = torch.mm(z1, z2.T)
        loss = self.contrast_loss(similarity, epoch)
        loss += self.beta*nn.MSELoss()(z1, z2)*self.n
        return loss
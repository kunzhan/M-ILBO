import argparse

from model import ENE
from dataset import load
from subset import subset_sampling, setup_seed
from loss import Compute_loss
from task import classfication
import torch
from time import *

import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Entropy Neural Estimation')

parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=1, help='GPU index.')
parser.add_argument('--epochs', type=int, default=40, help='Training epochs.')
parser.add_argument('--lr1', type=float, default=1e-3, help='Learning rate of M-ILBO.')
parser.add_argument('--lr2', type=float, default=1e-2, help='Learning rate of linear evaluator.')
parser.add_argument('--wd1', type=float, default=0, help='Weight decay of M-ILBO.')
parser.add_argument('--wd2', type=float, default=1e-4, help='Weight decay of linear evaluator.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
parser.add_argument('--ph', type=float, default=0.05, help='Drop feature ratio.')
parser.add_argument('--pa', type=float, default=0.3, help='Drop edge ratio.')
parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dim.')
parser.add_argument("--out_dim", type=int, default=512, help='Output layer dim.')
parser.add_argument('--top_k', type=int, default=10000, help='top k.')
parser.add_argument('--top_l', type=int, default=7320000, help='top l.') 
parser.add_argument('--labmda', type=float, default=0.35, help='lambda.') 
parser.add_argument('--use_mlp', action='store_true', default=False, help='Use MLP instead of GNN')

args = parser.parse_args()
# check cuda
if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

if __name__ == '__main__':
    
    begin_time = time()
    setup_seed(args.seed)

    graph, feat, labels, num_class, train_idx, val_idx, test_idx = load(args.dataname)
    print(args)
    in_dim = feat.shape[1]
    N = graph.number_of_nodes()

    model = ENE(in_dim, args.hid_dim, args.out_dim, args.n_layers)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    Loss = Compute_loss(args, N, graph)
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        graph1, feat1 = subset_sampling(graph, feat, args.ph, args.pa)
        graph2, feat2 = subset_sampling(graph, feat, args.ph, args.pa)

        graph1 = graph1.add_self_loop()
        graph2 = graph2.add_self_loop()

        graph1 = graph1.to(args.device)
        graph2 = graph2.to(args.device)

        feat1 = feat1.to(args.device)
        feat2 = feat2.to(args.device)

        z1, z2 = model(graph1, feat1, graph2, feat2)
        # z2 = (1-alpha)*z1.detach() +alpha*z2

        loss = Loss(epoch, z1, z2)
        loss.backward()
        optimizer.step()

        # print('\r\rEpoch={:03d}, loss={:.4f}'.format(epoch, loss.item()), end='  ')

    # print("\n=========================  Evaluation  =========================")
    # print('Epoch={:03d}'.format(epoch), end=' ')
    graph = graph.to(args.device)
    graph = graph.remove_self_loop().add_self_loop()
    feat = feat.to(args.device)

    embeds = model.get_embedding(graph, feat)
    classfication(args,embeds,labels,num_class,train_idx,val_idx,test_idx)
    # TSNE_plot(embeds.detach().cpu().numpy(), labels, args.dataname)
    end_time = time()
    run_time = end_time - begin_time
    print('Time:%5.4fs\n'%(run_time))
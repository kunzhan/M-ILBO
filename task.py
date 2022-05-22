
from model import LogReg
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from time import *
import os
import imageio
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ipdb import set_trace

def classfication(args,embeds,labels,num_class,train_idx,val_idx,test_idx,):
    train_embs = embeds[train_idx]
    val_embs = embeds[val_idx]
    test_embs = embeds[test_idx]

    label = labels.to(args.device)

    train_labels = label[train_idx]
    val_labels = label[val_idx]
    test_labels = label[test_idx]

    ''' Linear Evaluation '''
    logreg = LogReg(train_embs.shape[1], num_class)
    opt = torch.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)

    logreg = logreg.to(args.device)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    eval_acc = 0

    for epoch in range(2000):
        logreg.train()
        opt.zero_grad()
        logits = logreg(train_embs)
        preds = torch.argmax(logits, dim=1)
        train_acc = torch.sum(preds == train_labels).float() / train_labels.shape[0]
        loss = loss_fn(logits, train_labels)
        loss.backward()
        opt.step()

        logreg.eval()
        with torch.no_grad():
            val_logits = logreg(val_embs)
            test_logits = logreg(test_embs)

            val_preds = torch.argmax(val_logits, dim=1)
            test_preds = torch.argmax(test_logits, dim=1)

            val_acc = torch.sum(val_preds == val_labels).float() / val_labels.shape[0]
            test_acc = torch.sum(test_preds == test_labels).float() / test_labels.shape[0]

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                if test_acc > eval_acc:
                    eval_acc = test_acc

            # print('\r\rEpoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, train_acc, val_acc, test_acc), end=' ')

    print('  Linear evaluation accuracy:{:.4f}'.format(eval_acc))


class GIFPloter():
    def __init__(self, ):
        self.path_list = []

    def PlotOtherLayer(self,fig,data,label,title='',fig_position0=1,fig_position1=1,fig_position2=1,s=0.1,graph=None,link=None,):
        color_list = []
        for i in range(label.shape[0]):
            color_list.append(int(label[i]))

        if data.shape[1] > 3:
            pca = PCA(n_components=2)
            data_em = pca.fit_transform(data)
        else:
            data_em = data

        # data_em = data_em-data_em.mean(axis=0)

        if data_em.shape[1] == 3:
            ax = fig.add_subplot(fig_position0, fig_position1, fig_position2, projection='3d')

            ax.scatter(data_em[:, 0], data_em[:, 1], data_em[:, 2], c=color_list, s=s, cmap='rainbow')

        if data_em.shape[1] == 2:
            ax = fig.add_subplot(fig_position0, fig_position1, fig_position2)

            if graph is not None:
                self.PlotGraph(data, graph, link)

            s = ax.scatter(data_em[:, 0], data_em[:, 1], c=label, s=s, cmap='rainbow')
            plt.axis('equal')
            if None:
                list_i_n = len(set(label.tolist()))
                # print(list_i_n)
                legend1 = ax.legend(*s.legend_elements(num=list_i_n),
                                    loc="upper left",
                                    title="Ranking")
                ax.add_artist(legend1)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # plt.xticks([])
        # plt.yticks([])
        # plt.title(title)

    def AddNewFig(self,latent,label,link=None,graph=None,his_loss=None,title_='',path='./',dataset=None):
        fig = plt.figure(figsize=(5, 5))

        if latent.shape[0] <= 1000:   s=3
        elif latent.shape[0] <= 10000:   s = 1
        else:   s = 0.1

        # if latent.shape[1] <= 3:
        self.PlotOtherLayer(fig, latent, label, title=title_, fig_position0=1, fig_position1=1, fig_position2=1, graph=graph, link=link, s=s)
        plt.tight_layout()
        path_c = path + title_

        self.path_list.append(path_c)

        plt.savefig(path_c, dpi=100)
        plt.close()

    def PlotGraph(self, latent, graph, link):
        for i in range(graph.shape[0]):
            for j in range(graph.shape[0]):
                if graph[i, j] == True:
                    p1 = latent[i]
                    p2 = latent[j]
                    lik = link[i, j]
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]],
                            'gray',
                            lw=1 / lik)
                    if lik > link.min() * 1.01:
                        plt.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2,
                                str(lik)[:4],
                                fontsize=5)

    def SaveGIF(self):
        gif_images = []
        for i, path_ in enumerate(self.path_list):
            gif_images.append(imageio.imread(path_))
            if i > 0 and i < len(self.path_list)-2:
                os.remove(path_)
        imageio.mimsave(path_[:-4] + ".gif", gif_images, fps=3)

def TSNE_plot(X, label, str):
    em = TSNE(n_components=2,random_state=6).fit_transform(X)
    ploter = GIFPloter()
    ploter.AddNewFig(em, label, title_= str+".png", path='./figure/',)
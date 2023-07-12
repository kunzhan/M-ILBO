# Entropy Neural Estimation for Graph Contrastive Learning
Contrastive learning has demonstrated the success of the neural estimation of mutual information between high-dimensional data. We use contrastive loss to train a parameter-shared graph encoder by two sampled subsets of the raw dataset. In each epoch of the training process, the contrastive loss guides the encoder in estimating the mutual information between the two subsets. The up bound of their mutual information is the information entropy of the raw dataset since the two subsets are sampled from the same raw dataset. Data sampling is based on bootstrapping, so using different subsets renders the graph encoder to integrate information of the raw dataset. Besides, we modify the contrastive loss for enhancing the representation ability of the graph encoder. The two subsets can be seen as two views of the raw dataset, so the scores of positive and negative pairs can be selected in the cross-view similarity matrix. The classic positive pairs only locate at the main diagonal of the cross-view similarity matrix, we further add more positive pairs in non-diagonal whose scores are high, and negative pairs are those who have low scores. Extensive experiments on several benchmark datasets have demonstrated that the proposed information ensemble graph contrastive learning algorithm achieves the state-of-the-art performance. We also perform extensive ablation analysis to demonstrate the effectiveness of the proposed contrastive loss.

# Experiment
## Datasets
- Citation Networks: 'Cora', 'Citeseer' and 'Pubmed'.
- Co-occurence Networks: 'Amazon-Computer', 'Amazon-Photo', 'Coauthor-CS' and 'Coauthor-Physics'.

| Dataset          | # Nodes | # Edges | # Classes | # Features |
| ---------------- | ------- | ------- | --------- | ---------- |
| Cora             | 2,708   | 10,556  | 7         | 1,433      |
| Citeseer         | 3,327   | 9,228   | 6         | 3,703      |  
| Pubmed           | 19,717  | 88,651  | 3         | 500        |
| Amazon-Computer  | 13,752  | 574,418 | 10        | 767        |
| Amazon-Photo     | 7,650   | 287,326 | 8         | 745        |
| Coauthor-CS      | 18,333  | 327,576 | 15        | 6,805      |
| Coauthor-Physics | 34,493  | 991,848 | 5         | 8,451      |

## Usage
To run the codes, use the following commands:
```bash
pip install -r requirements.txt
```
```python
# Cora
python main.py --dataname cora --epochs 40 --dfr 0.05 --der 0.3 --beta 0.35 --neg_n 7320000 --wk 1000 --wb 8000

# Citeseer
python main.py --dataname citeseer --epochs 20 --n_layers 1 --dfr 0.0 --der 0.8 --wd2 1e-2 --beta 0.35 --neg_n 11055000 --wk 500 --wb 600

# Pubmed
python main.py --dataname pubmed --epochs 100 --der 0.979 --dfr 0.01 --hid_dim 128 --lr2 1e-2 --beta 0.2 --wk=1000 --wb=1000 --neg_n 388000000 --gpu -1

# Amazon-Computer
python main.py --dataname comp --epochs 100 --dfr 0.01 --der 0.01 --beta 0.05 --hid_dim 1024 --neg_n 188000000 --wk 500 --wb 10000 --gpu -1
python main.py --dataname comp --hid_dim 1024 --out_dim 1024 --use_mlp --epochs 15 --dfr 0.115 --neg_n 300000000 --beta 0.4 --wb 800

# Amazon-Photo
python main.py --dataname photo --epochs 50 --beta 0.1 --der 0.99 --dfr 0.1 --hid_dim 1024 --neg_n 58200000 --wk 5000 --wb 15000 

# Coauthor-CS
python main.py --dataname cs --hid_dim 1024 --out_dim 1024 --use_mlp --epochs 15 --dfr 0.115 --neg_n 300000000 --beta 0.4 --wb 5000 --wk 1000

# Coauthor-Physics
python main.py --dataname physics --epochs 100 --hid_dim 2048 --dfr 0.01 --der 0.01 --beta 0.4 --neg_n 800000000 --wk 1000 --wb 15000 --gpu -1
```
This code is heavily borrowed from the baseline [CCA-SSG](https://github.com/hengruizhang98/CCA-SSG)
# Citation
We appreciate it if you cite the following paper:
```
@InProceedings{MaNeurIPS2022,
  author =    {Yixuan Ma and Xiaolin Zhang and Peng Zhang and Kun Zhan},
  title =     {Entropy Neural Estimation for Graph Contrastive Learning},
  booktitle = {ACM MM},
  year =      {2023},
}

```

# Contact
https://kunzhan.github.io/

If you have any questions, feel free to contact me. (Email: `ice.echo#gmail.com`)
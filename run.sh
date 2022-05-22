# !/bin/bash

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
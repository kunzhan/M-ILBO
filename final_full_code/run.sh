# !/bin/bash
# Cora
python main.py

# python main.py --dataname citeseer \
# 				--epochs 20 \
# 				--n_layers 1 \
# 				--ph 0.0 \
# 				--pa 0.8 \
# 				--wd2 1e-2 \
# 				--top_l 11055000 \
# 				--top_k 6000


# python main.py --dataname pubmed \
# 				--epochs 100 \
# 				--hid_dim 128 \
# 				--pa 0.979 \
# 				--ph 0.01 \
# 				--lr2 1e-2 \
# 				--labmda 0.2 \
# 				--top_k 30000 \
# 				--top_l 388000000 \
# 				--gpu -1


# photo 9371
# python main.py --dataname photo \
# 				--epochs 50 \
# 				--labmda 0.1 \
# 				--ph 0.1 \
# 				--pa 0.99 \
# 				--hid_dim 1024 \
# 				--top_l 58200000 \
# 				--top_k 1000 \
# 				--gpu -1
# comp
# python main.py --dataname comp \
# 				--epochs 100  \
# 				--labmda 0.05 \
# 				--ph 0.01 \
# 				--pa 0.01 \
# 				--hid_dim 1024 \
# 				--top_l 188000000 \
# 				--top_k 100 \
# 				--gpu -1 \
# 				--lr1 1e-3 \
# 				--lr2 1e-2

# python main.py --dataname cs \
# 				 --hid_dim 1024 \
# 				 --out_dim 1024 \
# 				 --use_mlp \
# 				 --epochs 15 \
# 				 --ph 0.1 \
# 				 --pa 1 \
# 				 --top_l 300000000 \
# 				 --top_k 100 \
# 				 --labmda 0.4 \
# 				 --gpu -1 \
# 				 --lr1 1e-3 \
# 				 --lr2 1e-2

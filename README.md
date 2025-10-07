# Collaborative optimization of precision and diversity based on LightGCN recommendation system

what we do:

1. Add diverisity metric diversity, novelty and history deviation
2. Introducing diversity loss to dynamically adjust the trade-off between accuracy and diversity
3. Embedding hierarchical attention mechanism to assign weights to each layer of GCN and mining high-order neighbor information to enhance recommendation diversity

# Dataset

We provide two datasets: [LastFM](https://grouplens.org/datasets/hetrec-2011/) and [Ciao](https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.htm).

# Example to run the codes

1. Environment: I have tested this code with python3.8 Pytorch=1.7.1 CUDA=11.0
2. Run SocialLGN

   `python main.py --model=SocialLGN --dataset=lastfm --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --topks="[10,20]" --recdim=64 --bpr_batch=2048`

# Reference

LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

[SocialLGN: Light Graph Convolution Network for Social Recommendation](https://www.sciencedirect.com/science/article/abs/pii/S0020025522000019)

DGRec: Graph Neural Network for Recommendation with Diversified Embedding Generation

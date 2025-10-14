import world
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


class PureBPR(nn.Module):
    def __init__(self, config, dataset):
        super(PureBPR, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        print("using Normal distribution initializer")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss


class SubgraphSelection(nn.Module):
    def __init__(self, max_neighbors):
        super(SubgraphSelection, self).__init__()
        self.max_neighbors = max_neighbors
        self.cached_adj = None
        self.last_shape = None

    def forward(self, adj_matrix, embeddings):
        # Only resample if shape changes (e.g., at init or if graph changes)
        if self.cached_adj is not None and adj_matrix.size() == self.last_shape:
            return self.cached_adj

        device = embeddings.device
        indices = adj_matrix._indices().cpu().numpy()
        values = adj_matrix._values().cpu().numpy()
        num_nodes = adj_matrix.size(0)

        # Group neighbors by source node using numpy for speed
        from collections import defaultdict
        neighbors_dict = defaultdict(list)
        for src, dst, val in zip(indices[0], indices[1], values):
            neighbors_dict[src].append((dst, val))

        new_src = []
        new_dst = []
        new_val = []

        for src in range(num_nodes):
            neighbors = neighbors_dict.get(src, [])
            if len(neighbors) > self.max_neighbors:
                idx = np.random.choice(len(neighbors), self.max_neighbors, replace=False)
                sampled = [neighbors[i] for i in idx]
            else:
                sampled = neighbors
            for dst, val in sampled:
                new_src.append(src)
                new_dst.append(dst)
                new_val.append(val)

        if not new_src:
            self.cached_adj = adj_matrix
            self.last_shape = adj_matrix.size()
            return adj_matrix

        new_indices = torch.tensor([new_src, new_dst], dtype=torch.long, device=device)
        new_values = torch.tensor(new_val, dtype=torch.float, device=device)
        sampled_adj = torch.sparse.FloatTensor(new_indices, new_values, adj_matrix.size()).coalesce()
        self.cached_adj = sampled_adj
        self.last_shape = adj_matrix.size()
        return sampled_adj


class LightGCN(nn.Module):
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.layer_attention = config.get('layer_attention', 1)
        self.subgraph_selection = config.get('subgraph_selection', 1)
        self._init_weight()

    def _init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['layer']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()
        self.interactionGraph = self.dataset.getInteractionGraph()
        self.layer_attention = self.config.get('layer_attention', 1)
        if self.layer_attention:
            # 层注意力参数，初始化为均匀分布
            self.attention_weights = nn.Parameter(torch.ones(self.n_layers + 1) / (self.n_layers + 1))
        if self.subgraph_selection:
            self.neighbor_selector = SubgraphSelection(self.config['max_neighbors'])
        print(f"{world.model_name} is already to go")

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        G = self.interactionGraph
        if self.subgraph_selection:
            G = self.neighbor_selector(G, all_emb)

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(G, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)  # [num_nodes, n_layers+1, dim]
        if self.layer_attention:
            # softmax归一化
            attn = torch.softmax(self.attention_weights, dim=0)
            light_out = torch.sum(embs * attn.view(1, -1, 1), dim=1)
        else:
            light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        self.final_user, self.final_item = users, items
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.final_user, self.final_item
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss


class SocialLGN(LightGCN):
    def _init_weight(self):
        super(SocialLGN, self)._init_weight()
        self.socialGraph = self.dataset.getSocialGraph()
        self.Graph_Comb = Graph_Comb(self.latent_dim)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        A = self.interactionGraph
        S = self.socialGraph

        if self.subgraph_selection:
            A = self.neighbor_selector(A, all_emb)
            S = self.neighbor_selector(S, users_emb)

        embs = [all_emb]
        for layer in range(self.n_layers):
            users_emb, items_emb = torch.split(all_emb, [self.num_users, self.num_items])
            users_emb_social = torch.sparse.mm(S, users_emb)
            all_emb_interaction = torch.sparse.mm(A, all_emb)
            users_emb_interaction, items_emb_next = torch.split(all_emb_interaction, [self.num_users, self.num_items])
            users_emb_next = self.Graph_Comb(users_emb_social, users_emb_interaction)
            all_emb = torch.cat([users_emb_next, items_emb_next])
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        if self.layer_attention:
            attn = torch.softmax(self.attention_weights, dim=0)
            final_embs = torch.sum(embs * attn.view(1, -1, 1), dim=1)
        else:
            final_embs = torch.mean(embs, dim=1)
        users, items = torch.split(final_embs, [self.num_users, self.num_items])
        self.final_user, self.final_item = users, items
        return users, items


class Graph_Comb(nn.Module):
    def __init__(self, embed_dim):
        super(Graph_Comb, self).__init__()
        self.att_x = nn.Linear(embed_dim, embed_dim, bias=False)
        self.att_y = nn.Linear(embed_dim, embed_dim, bias=False)
        self.comb = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x, y):
        h1 = torch.tanh(self.att_x(x))
        h2 = torch.tanh(self.att_y(y))
        output = self.comb(torch.cat((h1, h2), dim=1))
        output = output / output.norm(2)
        return output

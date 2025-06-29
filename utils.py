import os
from time import time
from numpy.linalg import norm
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import optim

import world

class BPRLoss:
    def __init__(self, recmodel, config):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()

def UniformSample_original(users, dataset):
    total_start = time()
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S), [total, sample_time1, sample_time2]

# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName():
    if world.model_name == 'bpr':
        file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name in ['LightGCN', 'SocialLGN']:
        file = f"{world.model_name}-{world.dataset}-{world.config['layer']}layer-" \
               f"{world.config['latent_dim_rec']}.pth.tar"
    return os.path.join(world.FILE_PATH, file)

def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}

def MRRatK_r(r, k):
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def Diversity_atK(item_embeddings, recommended_items, k):
    """
    Compute diversity for a batch of recommendation lists.

    Args:
        item_embeddings: np.ndarray of shape (num_items, emb_dim)
        recommended_items: np.ndarray of shape (batch_size, k)
        k: int, number of recommended items for each user

    Returns:
        avg_diversity: float, average diversity over the batch
        diversities: np.ndarray of shape (batch_size,), diversity for each user
    """
    batch_size = recommended_items.shape[0]
    diversities = np.zeros(batch_size)
    for idx in range(batch_size):
        rec_ids = recommended_items[idx][:k]
        if len(rec_ids) <= 1:
            diversities[idx] = 0.0
            continue
        rec_embs = item_embeddings[rec_ids]
        # Normalize
        normed = rec_embs / np.linalg.norm(rec_embs, axis=1, keepdims=True)
        sims = np.dot(normed, normed.T)
        n = len(rec_ids)
        sim_sum = np.sum(np.triu(sims, 1))
        num_pairs = n * (n - 1) / 2
        avg_sim = sim_sum / num_pairs if num_pairs else 0.0
        diversities[idx] = 1 - avg_sim
    avg_diversity = np.mean(diversities)
    return avg_diversity, diversities
    
def HistoryDeviation(user_ids, rec_item_ids, dataset, recmodel, device):
    """
    计算历史偏离度（History Deviation），统一使用模型的 embedding_item 获取物品嵌入
    :param user_ids: 当前 batch 的用户 ID 列表，shape: [batch_size]
    :param rec_item_ids: 当前 batch 每个用户的推荐物品 ID 列表，shape: [batch_size, topk] 
    :param dataset: 数据集对象，用于获取用户历史交互物品
    :param recmodel: 推荐模型对象（PureBPR / LightGCN / SocialLGN）
    :param device: 计算设备（如 'cuda' 或 'cpu'）
    :return: 当前 batch 历史偏离度的均值，以及每个用户的偏离度列表
    """
    deviation_list = []
    
    for i, user_id in enumerate(user_ids):
        # 1. 获取用户历史兴趣向量：历史交互物品嵌入的均值
        user_hist_items = dataset.allPos[user_id]  # 用户历史交互物品 ID
        if len(user_hist_items) == 0:
            deviation_list.append(1.0)  # 无历史交互，偏离度设为 1
            continue
        
        # 2. 获取历史物品嵌入并计算均值
        hist_item_ids_tensor = torch.tensor(user_hist_items).long().to(device)
        hist_item_embs = recmodel.embedding_item(hist_item_ids_tensor).detach().cpu().numpy()
        user_hist_emb = np.mean(hist_item_embs, axis=0)  # 历史兴趣向量
        
        # 3. 获取推荐物品嵌入并计算均值
        rec_items = rec_item_ids[i]
        rec_item_ids_tensor = torch.tensor(rec_items).long().to(device)
        rec_item_embs = recmodel.embedding_item(rec_item_ids_tensor).detach().cpu().numpy()
        rec_agg_emb = np.mean(rec_item_embs, axis=0)  # 推荐聚合向量
        
        # 4. 计算余弦相似度和偏离度
        cos_sim = np.dot(user_hist_emb, rec_agg_emb) / (norm(user_hist_emb) * norm(rec_agg_emb)) \
            if (norm(user_hist_emb) * norm(rec_agg_emb)) != 0 else 0.0
        deviation = 1 - cos_sim
        deviation_list.append(deviation)
    
    mean_deviation = np.mean(deviation_list) if deviation_list else 0.0
    return mean_deviation, deviation_list




def AUC(all_item_scores, dataset, test_data):
    r_all = np.zeros((dataset.m_items,))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================
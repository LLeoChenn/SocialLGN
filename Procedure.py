import numpy as np
import torch

import utils
import world

# 新增：计算新颖性的函数
def calculateNovelty(dataset, recommendedItems):
    """
    计算推荐结果的新颖性
    :param dataset: 数据集对象
    :param recommendedItems: 推荐的物品列表，形状为 [用户数, topk]
    :return: 新颖性数值
    """
    noveltySum = 0
    n = 0
    for userItems in recommendedItems:
        for item in userItems:
            p_i = dataset.getItemPopularity(item)
            if p_i > 0:  
                # 避免对数无意义
                noveltySum += -np.log2(p_i)  # 使用log2更符合信息论定义
                n += 1
    if n == 0:
        return 0
    return noveltySum / n

def BPR_train_original(dataset, recommend_model, loss_class, epoch):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    allusers = list(range(dataset.n_users))
    S, sam_time = utils.UniformSample_original(allusers, dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
    aver_loss = aver_loss / total_batch
    return aver_loss

def test_one_batch(X, item_embeddings=None, k_list=None):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg, diversity = [], [], [], []
    for idx, k in enumerate(world.topks if k_list is None else k_list):
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
        if item_embeddings is not None:
            avg_div, _ = utils.Diversity_atK(item_embeddings, sorted_items, k)
            diversity.append(avg_div)
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg),
            'diversity': np.array(diversity) if diversity else None}

def mmr_rerank(item_embeddings, top_k_items, scores, lambda_mmr=0.5, T=2):
    """
    高效MMR重排序，使用numpy批量计算相似度，避免循环和dict。
    :param item_embeddings: Embeddings of all items.
    :param top_k_items: Top-K items predicted for a user.
    :param scores: Predicted scores for top_k_items (same order as top_k_items).
    :param lambda_mmr: Trade-off parameter between relevance and diversity.
    :param T: 筛选比例，输出K//T个物品
    :return: Re-ranked items (length K//T).
    """
    K = len(top_k_items)
    if K <= 1 or T < 1:
        return list(top_k_items)
    select_num = max(1, K // T)
    emb_matrix = item_embeddings[top_k_items]  # shape: [K, emb_dim]
    emb_norm = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8)
    sim_matrix = np.dot(emb_norm, emb_norm.T)  # shape: [K, K]
    np.fill_diagonal(sim_matrix, 0.0)
    selected = []
    candidate = list(range(K))
    scores = np.array(scores)
    while len(selected) < select_num:
        if not selected:
            idx = np.argmax(scores[candidate])
        else:
            max_sim = sim_matrix[candidate][:, selected].max(axis=1) if selected else np.zeros(len(candidate))
            mmr_score = lambda_mmr * scores[candidate] - (1 - lambda_mmr) * max_sim
            idx = np.argmax(mmr_score)
        selected.append(candidate[idx])
        candidate.pop(idx)
    return [top_k_items[i] for i in selected]

def Test(dataset, Recmodel, epoch, cold=False, w=None):
    u_batch_size = world.config['test_u_batch_size']
    if cold:
        testDict: dict = dataset.coldTestDict
    else:
        testDict: dict = dataset.testDict
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks)),
               'diversity': np.zeros(len(world.topks)),
               'history_deviation': 0.0,  # 新增历史偏离度指标
               'novelty': np.zeros(len(world.topks))}  # 新增新颖性指标

    recommendedItemsList = []

    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        total_history_deviation = 0.0

        # get embeddings for diversity
        if hasattr(Recmodel, "final_item"):
            item_embeddings = Recmodel.final_item.cpu().detach().numpy()
        elif hasattr(Recmodel, "embedding_item"):
            item_embeddings = Recmodel.embedding_item.weight.cpu().detach().numpy()
        else:
            item_embeddings = None

        lambda_mmr = world.config['lambda_mmr'] if item_embeddings is not None else None
        mmr_T = world.config.get('mmr_T', 2)
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            # 先取max_K个候选
            scores, rating_K = torch.topk(rating, k=max_K)
            # MMR重排序（如果需要）
            if item_embeddings is not None and lambda_mmr is not None and lambda_mmr < 1.0:
                reranked_items = []
                reranked_scores = []
                rating_K_np = rating_K.cpu().numpy()
                scores_np = scores.cpu().numpy()
                for user_idx in range(rating_K_np.shape[0]):
                    top_k_items = rating_K_np[user_idx]
                    top_k_scores = scores_np[user_idx]
                    mmr_items = mmr_rerank(item_embeddings, top_k_items, top_k_scores, lambda_mmr=lambda_mmr, T=mmr_T)
                    reranked_items.append(mmr_items)
                    reranked_scores.append([top_k_scores[list(top_k_items).index(i)] for i in mmr_items])
                rating_K = torch.tensor(reranked_items)
                scores = torch.tensor(reranked_scores)
                used_K = max(1, max_K // mmr_T)
            else:
                used_K = max_K
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
            # 只保留推荐物品id，减少内存压力
            recommendedItemsList.extend(rating_K.cpu().numpy().tolist())
            batch_rec_items = rating_K.cpu().numpy()
            batch_deviation, _ = utils.HistoryDeviation(
                batch_users, 
                batch_rec_items, 
                dataset, 
                Recmodel,
                world.device
            )
            total_history_deviation += batch_deviation * len(batch_users)

        assert total_batch == len(users_list)
        # 指标计算时，K变为used_K
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            # 传递used_K，确保指标计算与推荐序列长度一致
            pre_results.append(test_one_batch(x, item_embeddings, k_list=[used_K for _ in world.topks]))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            if result['diversity'] is not None:
                results['diversity'] += result['diversity']

        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        results['diversity'] /= float(len(users))
        results['history_deviation'] = total_history_deviation / float(len(users))

        # 新颖性指标
        novelty = calculateNovelty(dataset, recommendedItemsList)
        results['novelty'] = np.array([novelty] * len(world.topks))  

        print(f"新颖性（Novelty）: {novelty}")

        return results
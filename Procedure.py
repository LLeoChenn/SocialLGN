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
    
    # 新增：用于收集推荐的物品，后续计算新颖性
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
            _, rating_K = torch.topk(rating, k=max_K)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
            
            # 新增：收集推荐的物品
            recommendedItemsList.extend(rating_K.cpu().numpy())  
            
            batch_rec_items = rating_K.cpu().numpy()  # 推荐物品ID
            batch_deviation, _ = utils.HistoryDeviation(
                batch_users, 
                batch_rec_items, 
                dataset, 
                Recmodel,
                world.device
            )
            total_history_deviation += batch_deviation * len(batch_users)  # 加权求和
        
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x, item_embeddings))
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
        
        # 新增：计算并输出新颖性
        novelty = calculateNovelty(dataset, recommendedItemsList)
        results['novelty'] = np.array([novelty] * len(world.topks))  
        
        print(f"新颖性（Novelty）: {novelty}")
        print(results)

        return results
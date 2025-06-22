import time
from os.path import join

import torch
import pandas as pd

import Procedure
import register
import utils
import world
from register import dataset

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
torch.autograd.set_detect_anomaly(True)
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        print(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

# 创建DataFrame来存储训练信息
columns = ['epoch', 'loss', 'recall', 'ndcg', 'precision', 'diversity',
           'recall_cold', 'ndcg_cold', 'precision_cold', 'diversity_cold',
           'novelty', 'novelty_cold',
           'history_deviation', 'history_deviation_cold']
train_history = pd.DataFrame(columns=columns)

best_ndcg, best_recall, best_pre, best_diversity = 0, 0, 0, 0
best_ndcg_cold, best_recall_cold, best_pre_cold, best_diversity_cold = 0, 0, 0, 0
best_deviation, best_deviation_cold = 1.0, 1.0  # 初始化历史偏离度最佳值
low_count, low_count_cold = 0, 0
try:
    for epoch in range(world.TRAIN_epochs + 1):
        print('======================')
        print(f'EPOCH[{epoch}/{world.TRAIN_epochs}]')
        start = time.time()

        # 初始化当前epoch的记录
        epoch_data = {'epoch': epoch, 'loss': None, 'recall': None, 'ndcg': None, 'precision': None,'diversity': None,
                       'recall_cold': None, 'ndcg_cold': None, 'precision_cold': None, 
                       'diversity_cold': None,
                       'novelty': None, 'novelty_cold': None,
                       'history_deviation': None,  # 新增
                       'history_deviation_cold': None  # 新增
                       }

        if epoch % 10 == 1 or epoch == world.TRAIN_epochs:
            print("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch, False)
            results_cold = Procedure.Test(dataset, Recmodel, epoch, True)

            # 保存测试结果到DataFrame
            epoch_data['recall'] = results['recall'][0]
            epoch_data['ndcg'] = results['ndcg'][0]
            epoch_data['precision'] = results['precision'][0]
            epoch_data['diversity'] = results['diversity'][0] if results['diversity'] is not None else None
            epoch_data['novelty'] = results['novelty'][0]  # 新增：保存正常测试的新颖性
            epoch_data['history_deviation'] = results['history_deviation']  # 新增

            epoch_data['recall_cold'] = results_cold['recall'][0]
            epoch_data['ndcg_cold'] = results_cold['ndcg'][0]
            epoch_data['precision_cold'] = results_cold['precision'][0]
            epoch_data['diversity_cold'] = results_cold['diversity'][0] if results_cold['diversity'] is not None else None
            epoch_data['novelty_cold'] = results_cold['novelty'][0]  # 新增：保存冷启动测试的新颖性
            epoch_data['history_deviation_cold'] = results_cold['history_deviation']  # 新增

            if results['ndcg'][0] < best_ndcg:
                low_count += 1
                if low_count == 30:
                    if epoch > 1000:
                        break
                    else:
                        low_count = 0
            else:
                best_recall = results['recall'][0]
                best_ndcg = results['ndcg'][0]
                best_pre = results['precision'][0]
                best_diversity = results['diversity'][0]
                best_novelty = results['novelty'][0]
                best_history_deviation = results['history_deviation']
                low_count = 0

            if results_cold['ndcg'][0] > best_ndcg_cold:
                best_recall_cold = results_cold['recall'][0]
                best_ndcg_cold = results_cold['ndcg'][0]
                best_pre_cold = results_cold['precision'][0]
                best_diversity_cold = results_cold['diversity'][0]
                best_novelty_cold = results_cold['novelty'][0]
                best_history_deviation_cold = results_cold['history_deviation']
                low_count_cold = 0

        loss = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch)
        print(f'[saved][BPR aver loss{loss:.3e}]')

        # 保存损失值到DataFrame
        epoch_data['loss'] = loss
        # 将当前epoch的数据添加到DataFrame
        train_history = pd.concat([train_history, pd.DataFrame([epoch_data])], ignore_index=True)

        torch.save(Recmodel.state_dict(), weight_file)
finally:
    print(f"\nbest recall at 10:{best_recall}")
    print(f"best ndcg at 10:{best_ndcg}")
    print(f"best precision at 10:{best_pre}")
    print(f"best diversity at 10:{best_diversity}")
    print(f"best novelty at 10:{best_novelty}")
    print(f"best history_deviation at 10:{best_history_deviation}")
    print(f"\nbest recall at 10:{best_recall_cold}")
    print(f"best ndcg at 10:{best_ndcg_cold}")
    print(f"best precision at 10:{best_pre_cold}")
    print(f"best diversity at 10:{best_diversity_cold}")
    print(f"best novelty at 10:{best_novelty_cold}")
    print(f"best history_deviation at 10:{best_history_deviation_cold}")
    
    # 保存训练历史到CSV文件
    # history_file = weight_file.replace('.pth', '_history_test.csv')
    history_file = 'test.csv'
    train_history.to_csv(history_file,mode='a', index=False)
    print(f"\n训练历史已保存到: {history_file}")  
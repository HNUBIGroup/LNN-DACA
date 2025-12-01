# 导入DaLPI
from model import DaLNPI
from utile import *
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import os


def set_seed(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(2022)
warnings.filterwarnings("ignore")

# -------------------------- 参数配置 --------------------------
# RPI488 emb=2 5e-4  weight_decay=1e-4 五折
# NPInter2 emb=4 5e-4 weight_decay=1e-6 10折
# RPI1807 emb=4 1e-4 weight_decay=1e-6  10折
# RPI2213 emb=4 1e-4 weight_decay=1e-6  10折

batch_size = 1024
embedding_size = 4
data_name = 'RPI1807'
epoches =100
n_splits = 10

# 关键改动：指定预训练权重的路径 (此行已删除)
# WARMUP_WEIGHTS_PATH = f'dalnpi_warmup_weights_{data_name}.pt'

# -------------------------- 数据加载与预处理 (与原版一致) --------------------------
sparse_feature = ['R' + str(i) for i in range(1, 257)]
dense_feature = ['P' + str(i) for i in range(1, 401)]
col_names = ['label'] + dense_feature + sparse_feature
data = pd.read_csv(f'data/{data_name}/sample.txt', names=col_names, sep='\t')
data[sparse_feature] = data[sparse_feature].fillna('-1')
data[dense_feature] = data[dense_feature].fillna('0')
feat_sizes = {feat: len(data[feat].unique()) for feat in dense_feature + sparse_feature}
for feat in sparse_feature:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
nms = MinMaxScaler(feature_range=(0, 1))
data[dense_feature] = nms.fit_transform(data[dense_feature])
fixlen_feature_columns = [(feat, 'sparse') for feat in sparse_feature] + [(feat, 'dense') for feat in dense_feature]
dnn_feature_columns = fixlen_feature_columns

# -------------------------- K-Fold 交叉验证 --------------------------
kf = KFold(n_splits=n_splits, shuffle=True, random_state=2022)
all_fold_metrics = []

for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
    print(f"\n===== 第 {fold + 1}/{n_splits} 折训练 =====")

    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()
    train_label = pd.DataFrame(train['label'])
    train_features = train.drop(columns=['label'])
    test_label = pd.DataFrame(test['label'])
    test_features = test.drop(columns=['label'])
    train_tensor = TensorDataset(torch.from_numpy(np.array(train_features)), torch.from_numpy(np.array(train_label)))
    test_tensor = TensorDataset(torch.from_numpy(np.array(test_features)), torch.from_numpy(np.array(test_label)))
    train_loader = DataLoader(train_tensor, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_tensor, batch_size=batch_size)

    model = DaLNPI(feat_sizes, embedding_size, dnn_feature_columns).cuda()

    # --- 关键改动：删除了加载预训练权重的 if/else 逻辑块 ---
    print("  模型已随机初始化，将从头开始训练。")

    loss_func = nn.BCELoss(reduction='mean').cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

    fold_best_metrics = {
        'auc': {'value': -1, 'epoch': -1}, 'sen': {'value': -1, 'epoch': -1},
        'pre': {'value': -1, 'epoch': -1}, 'f1': {'value': -1, 'epoch': -1},
        'acc': {'value': -1, 'epoch': -1}, 'spe': {'value': -1, 'epoch': -1},
        'mcc': {'value': -1, 'epoch': -1}
    }

    for epoch in range(epoches):
        total_loss = 0.0
        model.train()
        for x, y in train_loader:
            x, y = x.cuda().float(), y.cuda().float()
            y_hat = model(x)
            optimizer.zero_grad()
            loss = loss_func(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        auc, sen, pre, f1, acc, spe, mcc = get_result(test_loader, model)
        avg_loss = total_loss / len(train_loader)
        print(f"折 {fold + 1} - 轮次 {epoch}/{epoches} | 训练损失: {avg_loss:.5f} | "
              f"AUC: {auc:.6f}, SEN: {sen:.6f}, PRE: {pre:.6f}, F1: {f1:.6f}, "
              f"ACC: {acc:.6f}, SPE: {spe:.6f}, MCC: {mcc:.6f}")

        # 更新最佳指标逻辑
        if not np.isclose(auc, 1.0, atol=1e-6) and auc > fold_best_metrics['auc']['value']:
            fold_best_metrics['auc'] = {'value': auc, 'epoch': epoch}
            # =======================================================================
            # 新增改动：在这里保存当前折的最佳模型
            # 1. 定义您指定的保存目录 (使用 r'' 原始字符串以避免路径错误)
            save_dir = fr'D:\XiaZai\MHAM-NPI-main\MHAM-NPI-main\data\{data_name}'

            # 2. 确保这个目录存在，如果不存在就创建它
            os.makedirs(save_dir, exist_ok=True)

            # 3. 组合目录和文件名，创建完整路径
            filename = f'dalnpi_best_model_fold_{fold + 1}.pt'
            BEST_MODEL_SAVE_PATH = os.path.join(save_dir, filename)

            # 4. 保存模型
            torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
            print(f"  *** 发现更优AUC，已保存模型到 {BEST_MODEL_SAVE_PATH} ***")
            # =======================================================================

        # (其他指标的更新逻辑保持不变)
        if not np.isclose(sen, 1.0, atol=1e-6) and sen > fold_best_metrics['sen']['value']:
            fold_best_metrics['sen'] = {'value': sen, 'epoch': epoch}
        if not np.isclose(pre, 1.0, atol=1e-6) and pre > fold_best_metrics['pre']['value']:
            fold_best_metrics['pre'] = {'value': pre, 'epoch': epoch}
        if not np.isclose(f1, 1.0, atol=1e-6) and f1 > fold_best_metrics['f1']['value']:
            fold_best_metrics['f1'] = {'value': f1, 'epoch': epoch}
        if not np.isclose(acc, 1.0, atol=1e-6) and acc > fold_best_metrics['acc']['value']:
            fold_best_metrics['acc'] = {'value': acc, 'epoch': epoch}
        if not np.isclose(spe, 1.0, atol=1e-6) and spe > fold_best_metrics['spe']['value']:
            fold_best_metrics['spe'] = {'value': spe, 'epoch': epoch}
        if not np.isclose(mcc, 1.0, atol=1e-6) and mcc > fold_best_metrics['mcc']['value']:
            fold_best_metrics['mcc'] = {'value': mcc, 'epoch': epoch}

    all_fold_metrics.append(fold_best_metrics)

    print(f"\n--- 第 {fold + 1} 折最佳结果 ---")
    for key in fold_best_metrics:
        value = fold_best_metrics[key]['value']
        epoch_num = fold_best_metrics[key]['epoch']
        print(f"  最佳 {key.upper()}: {value:.6f} (在第 {epoch_num} 轮)")


print(f"\n===== 所有 {n_splits} 折交叉验证最终平均结果 =====")

# 计算各项指标的平均值
avg_auc = np.mean([m['auc']['value'] for m in all_fold_metrics])
avg_sen = np.mean([m['sen']['value'] for m in all_fold_metrics])
avg_pre = np.mean([m['pre']['value'] for m in all_fold_metrics])
avg_f1 = np.mean([m['f1']['value'] for m in all_fold_metrics])
avg_acc = np.mean([m['acc']['value'] for m in all_fold_metrics])
avg_spe = np.mean([m['spe']['value'] for m in all_fold_metrics])
avg_mcc = np.mean([m['mcc']['value'] for m in all_fold_metrics])

# 打印平均结果
print(f"平均 AUC: {avg_auc:.6f}")
print(f"平均 SEN: {avg_sen:.6f}")
print(f"平均 PRE: {avg_pre:.6f}")
print(f"平均 F1: {avg_f1:.6f}")
print(f"平均 ACC: {avg_acc:.6f}")
print(f"平均 SPE: {avg_spe:.6f}")
print(f"平均 MCC: {avg_mcc:.6f}")
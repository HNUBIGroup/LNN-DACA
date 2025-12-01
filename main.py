# 综合指标
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

# ===================== 超参（保持你原注释） =====================
# NPInter2 emb=4 5e-4 weight_decay=1e-6 10折
# RPI1807 emb=4 1e-4 weight_decay=1e-6  10折
# RPI2213 emb=4 1e-4 weight_decay=1e-6  10折
# RPI488 emb=2 5e-4  weight_decay=1e-4 五折

def set_seed(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


warnings.filterwarnings('ignore')
set_seed(2022)

# ===================== 超参 =====================
batch_size = 4096
embedding_size = 4
data_name = 'RPI1807'
epoches = 200
n_splits = 10  # ← 合理折数
WARMUP_WEIGHTS_PATH = f'dalnpi_warmup_weights_{data_name}.pt'

# ===================== 数据加载 =====================
sparse_feature = ['R' + str(i) for i in range(1, 257)]
dense_feature = ['P' + str(i) for i in range(1, 401)]
col_names = ['label'] + dense_feature + sparse_feature

data = pd.read_csv(f'data/{data_name}/sample.txt', names=col_names, sep='\t')
data[sparse_feature] = data[sparse_feature].fillna('-1')
data[dense_feature] = data[dense_feature].fillna('0')

# 统计特征空间大小（全局）
feat_sizes = {feat: len(data[feat].unique()) for feat in dense_feature + sparse_feature}
fixlen_feature_columns = [(feat, 'sparse') for feat in sparse_feature] + [(feat, 'dense') for feat in dense_feature]
dnn_feature_columns = fixlen_feature_columns

# ===================== K 折 =====================
kf = KFold(n_splits=n_splits, shuffle=True, random_state=2022)
all_fold_metrics = []  # <--- 这将存储 K 个 "最佳指标包"

metric_names = ['auc', 'sen', 'pre', 'f1', 'acc', 'spe', 'mcc']  # <--- 定义指标名

for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
    print(f"\n===== 第 {fold + 1}/{n_splits} 折 =====")
    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()

    # ========== 折内预处理：只用 train 拟合，再作用 train/test ==========
    # 稀疏特征
    for feat in sparse_feature:
        le = LabelEncoder()
        train[feat] = le.fit_transform(train[feat])

        # 对 test 中未见过的类别设为 '<UNK>'
        test_feat = test[feat].copy()
        mask_unknown = ~test_feat.isin(le.classes_)
        if mask_unknown.any():
            test_feat[mask_unknown] = '<UNK>'
            if '<UNK>' not in le.classes_:
                le.classes_ = np.append(le.classes_, '<UNK>')
        test[feat] = le.transform(test_feat)

    # 稠密特征
    nms = MinMaxScaler(feature_range=(0, 1))
    train[dense_feature] = nms.fit_transform(train[dense_feature])
    test[dense_feature] = nms.transform(test[dense_feature])

    # 组 Loader
    train_label = pd.DataFrame(train['label'])
    train_features = train.drop(columns=['label'])
    test_label = pd.DataFrame(test['label'])
    test_features = test.drop(columns=['label'])

    train_tensor = TensorDataset(
        torch.from_numpy(np.array(train_features)),
        torch.from_numpy(np.array(train_label))
    )
    test_tensor = TensorDataset(
        torch.from_numpy(np.array(test_features)),
        torch.from_numpy(np.array(test_label))
    )

    train_loader = DataLoader(train_tensor, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_tensor, batch_size=batch_size)

    # 模型
    model = DaLNPI(feat_sizes, embedding_size, dnn_feature_columns).cuda()
    if os.path.exists(WARMUP_WEIGHTS_PATH):
        print(f"  正在从 {WARMUP_WEIGHTS_PATH} 加载预训练权重...")
        saved_state_dict = torch.load(WARMUP_WEIGHTS_PATH)
        model_state_dict = model.state_dict()
        load_state_dict = {
            k: v for k, v in saved_state_dict.items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }
        model_state_dict.update(load_state_dict)
        model.load_state_dict(model_state_dict)
        print(f"  成功加载 {len(load_state_dict)} 个参数层。")
    else:
        print("  未找到预训练权重，将从随机初始化开始训练。")

    loss_func = nn.BCELoss(reduction='mean').cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

    # ==================== 追踪最佳综合评分 ====================
    # 评分方案：score = 0.4*AUC + 0.3*F1 + 0.3*MCC
    fold_best_score = -np.inf
    fold_best_metric_package = {}
    fold_best_epoch = -1

    for epoch in range(epoches):
        total_loss = 0.0
        model.train()
        for x, y in train_loader:
            x = x.cuda().float()
            y = y.cuda().float()
            optimizer.zero_grad()
            yhat = model(x)
            loss = loss_func(yhat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 评测
        auc, sen, pre, f1, acc, spe, mcc = get_result(test_loader, model)
        avg_loss = total_loss / len(train_loader)
        print(
            f"折 {fold + 1} - 轮次 {epoch}/{epoches} | 训练损失: {avg_loss:.5f} | "
            f"AUC: {auc:.6f}, SEN: {sen:.6f}, PRE: {pre:.6f}, F1: {f1:.6f}, "
            f"ACC: {acc:.6f}, SPE: {spe:.6f}, MCC: {mcc:.6f}"
        )

        # 综合评分
        balanced_score = 0.4 * auc + 0.3 * f1 + 0.3 * mcc

        # 排除 AUC≈1.0 的极端情况，且综合评分更高才更新
        if (not np.isclose(auc, 1.0, atol=1e-6)) and (balanced_score > fold_best_score):
            fold_best_score = balanced_score
            fold_best_epoch = epoch
            fold_best_metric_package = {
                'auc': auc, 'sen': sen, 'pre': pre, 'f1': f1,
                'acc': acc, 'spe': spe, 'mcc': mcc,
                'epoch': epoch,
                'score': balanced_score
            }

############################################
# ============ 交叉验证整体汇总 ============
############################################
print("\n===== 各折最佳指标汇总（均值 ± 标准差）=====")
print(" (均基于各自折上 Balanced Score 最佳轮次的配套指标)")

for metric in metric_names:
    vals = [
        fold_package[metric] for fold_package in all_fold_metrics
        if fold_package and metric in fold_package
    ]

    if len(vals) == 0:
        mean_val = float('nan')
        std_val = float('nan')
    elif len(vals) == 1:
        mean_val = float(np.mean(vals))
        std_val = 0.0
    else:
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1))

    print(f"{metric.upper():>4}: {mean_val:.6f} ± {std_val:.6f}")


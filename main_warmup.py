# 导入DALNPI_warmup
from model import DaLNPI_warmup
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

def set_seed(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

warnings.filterwarnings('ignore')
set_seed(2022)

# ===================== 超参（保持你原注释） =====================
# NPInter2 emb=4 5e-4 weight_decay=1e-6 10折
# RPI1807 emb=4 1e-4 weight_decay=1e-6  10折
# RPI2213 emb=4 1e-4 weight_decay=1e-6  10折
# RPI488 emb=2 5e-4  weight_decay=1e-4 五折

batch_size = 4096
embedding_size = 4
data_name = 'RPI1807'
epoches = 200
n_splits = 10  # ← 合理折数
WARMUP_WEIGHTS_PATH = f'dalnpi_warmup_weights_{data_name}.pt'

# ===================== 数据加载（只做缺失处理，不做全量拟合） =====================
sparse_feature = ['R' + str(i) for i in range(1, 257)]
dense_feature  = ['P' + str(i) for i in range(1, 401)]
col_names = ['label'] + dense_feature + sparse_feature

data = pd.read_csv(f'data/{data_name}/sample.txt', names=col_names, sep='\t')
data[sparse_feature] = data[sparse_feature].fillna('-1')
data[dense_feature]  = data[dense_feature].fillna('0')

# 先基于原始数据统计特征空间大小（用于模型 embedding 初始化）
feat_sizes = {feat: len(data[feat].unique()) for feat in dense_feature + sparse_feature}
fixlen_feature_columns = [(feat, 'sparse') for feat in sparse_feature] + [(feat, 'dense') for feat in dense_feature]
dnn_feature_columns = fixlen_feature_columns

# ===================== 只用第一折做 warmup =====================
kf = KFold(n_splits=n_splits, shuffle=True, random_state=2022)
print("===== 开始预训练 (使用第一折数据) =====")
train_idx, _ = next(iter(kf.split(data)))
train = data.iloc[train_idx].copy()

# 在训练集中再切出 10% 作为 warmup 验证
train = train.sample(frac=1, random_state=2022).reset_index(drop=True)
val_split = int(len(train) * 0.9)
val = train.iloc[val_split:].copy()
train = train.iloc[:val_split].copy()

# ===================== 折内预处理：只用 train 拟合，再作用 train/val =====================
# 稀疏特征（LabelEncoder）
for feat in sparse_feature:
    le = LabelEncoder()
    train[feat] = le.fit_transform(train[feat])

    # 未见类别回退到 <UNK>
    val[feat] = val[feat].where(val[feat].isin(le.classes_), '<UNK>')
    if '<UNK>' not in le.classes_:
        le.classes_ = np.append(le.classes_, '<UNK>')
    val[feat] = le.transform(val[feat])

# 稠密特征（MinMaxScaler）
nms = MinMaxScaler(feature_range=(0, 1))
train[dense_feature] = nms.fit_transform(train[dense_feature])
val[dense_feature]   = nms.transform(val[dense_feature])

# ===================== 组 Loader，开训 =====================
train_label = pd.DataFrame(train['label'])
train_features = train.drop(columns=['label'])
val_label = pd.DataFrame(val['label'])
val_features = val.drop(columns=['label'])

train_tensor = TensorDataset(torch.from_numpy(np.array(train_features)),
                             torch.from_numpy(np.array(train_label)))
val_tensor   = TensorDataset(torch.from_numpy(np.array(val_features)),
                             torch.from_numpy(np.array(val_label)))

train_loader = DataLoader(train_tensor, shuffle=True, batch_size=batch_size)
val_loader   = DataLoader(val_tensor, batch_size=batch_size)

model = DaLNPI_warmup(feat_sizes, embedding_size, dnn_feature_columns).cuda()
loss_func = nn.BCELoss(reduction='mean').cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

best_val_auc = -1
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

    # 验证
    if len(val_loader) == 0:
        print(f"预训练 - 轮次 {epoch}/{epoches} | 训练损失: {total_loss / len(train_loader):.5f} | 验证集为空，跳过评估")
        continue

    auc, sen, pre, f1, acc, spe, mcc = get_result(val_loader, model)
    avg_loss = total_loss / len(train_loader)
    print(f"预训练 - 轮次 {epoch}/{epoches} | 训练损失: {avg_loss:.5f} | 验证 AUC: {auc:.6f}")

    if auc > best_val_auc:
        best_val_auc = auc
        torch.save(model.state_dict(), WARMUP_WEIGHTS_PATH)
        print(f"  * 发现更优模型，已保存权重到 {WARMUP_WEIGHTS_PATH}")

print("\n===== 预训练完成 =====")
print(f"最优模型权重已保存在: {WARMUP_WEIGHTS_PATH}")

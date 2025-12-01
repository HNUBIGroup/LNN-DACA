import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import DaLNPI
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os


# -------------------------- 参数配置 --------------------------
embedding_size = 4
data_name = 'RPI1807'  # <<-- 从此变量自动获取
n_splits = 10


# -------------------------- 2. 数据加载与预处理 --------------------------
sparse_feature = ['R' + str(i) for i in range(1, 257)]
dense_feature = ['P' + str(i) for i in range(1, 401)]
col_names = ['label'] + dense_feature + sparse_feature

data = pd.read_csv(f'data/{data_name}/sample.txt', names=col_names, sep='\t')

data[sparse_feature] = data[sparse_feature].fillna('-1')
data[dense_feature] = data[dense_feature].fillna('0')

feat_sizes = {feat: len(data[feat].unique()) for feat in dense_feature + sparse_feature}

lbe_dic = {}
for feat in sparse_feature:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
    lbe_dic[feat] = lbe

nms = MinMaxScaler(feature_range=(0, 1))
data[dense_feature] = nms.fit_transform(data[dense_feature])

fixlen_feature_columns = [(feat, 'sparse') for feat in sparse_feature] + \
                         [(feat, 'dense') for feat in dense_feature]
dnn_feature_columns = fixlen_feature_columns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------- 3. 选择一个样本进行解释 --------------------------
positive_samples = data[data['label'] == 1]
sample_index = positive_samples.index[0]
sample_row = positive_samples.loc[sample_index]
sample_features = sample_row.drop('label')

input_tensor = torch.from_numpy(np.array(sample_features.values)).unsqueeze(0).to(device).float()
input_tensor.requires_grad = True


# -------------------------- 4. 遍历十折模型，计算梯度重要性 --------------------------
protein_scores_all_folds = []
rna_scores_all_folds = []

print(f"开始对 {n_splits} 折的模型进行定量平均分析...")

for fold in range(1, n_splits + 1):

    model_dir = fr'D:\XiaZai\MHAM-NPI-main\MHAM-NPI-main\data\{data_name}'
    filename = f'dalnpi_best_model_fold_{fold}.pt'
    model_path = os.path.join(model_dir, filename)

    if not os.path.exists(model_path):
        print(f"警告：找不到模型文件 {model_path}，已跳过。")
        continue

    print(f"  正在分析第 {fold}/{n_splits} 折的模型: {model_path}")

    model = DaLNPI(feat_sizes, embedding_size, dnn_feature_columns).to(device)
    # model = DaLNPI(feat_sizes, embedding_size, dnn_feature_columns, att_layer_num=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if input_tensor.grad is not None:
        input_tensor.grad.zero_()

    output = model(input_tensor)
    model.zero_grad()
    output.backward()

    # Protein 特征梯度
    protein_importance = input_tensor.grad.abs().squeeze(0).cpu().numpy()[:len(dense_feature)]
    protein_scores_all_folds.append(protein_importance)

    # RNA 特征梯度
    rna_importance = np.zeros(len(sparse_feature))
    sparse_feature_names = [feat for feat, t in dnn_feature_columns if t == 'sparse']

    for i, feat_name in enumerate(sparse_feature_names):
        index_val = int(sample_row[feat_name])
        embedding_layer = model.embedding_dic[feat_name]

        if embedding_layer.weight.grad is not None:
            grad_for_index = embedding_layer.weight.grad[index_val]
            rna_importance[i] = torch.norm(grad_for_index).item()

    rna_scores_all_folds.append(rna_importance)

print("所有模型分析完毕！开始计算平均值和标准差...")


# -------------------------- 5. 计算平均值与标准差 --------------------------
avg_protein_importance = np.mean(protein_scores_all_folds, axis=0)
std_protein_importance = np.std(protein_scores_all_folds, axis=0)

avg_rna_importance = np.mean(rna_scores_all_folds, axis=0)
std_rna_importance = np.std(rna_scores_all_folds, axis=0)


# -------------------------- 6. 绘图（蓝橙配色） --------------------------
plt.style.use('seaborn-v0_8-whitegrid')

protein_color = "#1f77b4"  # 蓝色
rna_color = "#ff7f0e"      # 橙色
err_color = "#444444"      # 误差线

fig, axes = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={'hspace': 0.35})

fig.suptitle(f'Feature Importance Analysis on {data_name} Dataset',
             fontsize=22, fontweight='bold')


# -------- Protein --------
axes[0].bar(
    range(len(avg_protein_importance)),
    avg_protein_importance,
    color=protein_color,
    alpha=0.85
)
axes[0].errorbar(
    range(len(avg_protein_importance)),
    avg_protein_importance,
    yerr=std_protein_importance,
    fmt='none',
    ecolor=err_color,
    alpha=0.3
)

axes[0].set_title(f'Average Protein Feature Importance ({n_splits}-Fold CV)', fontsize=18)
axes[0].set_xlabel('Feature Index (P1 to P400)', fontsize=14)
axes[0].set_ylabel('Average Gradient Magnitude', fontsize=14)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].set_xlim(-10, len(avg_protein_importance) + 10)


# -------- RNA --------
axes[1].bar(
    range(len(avg_rna_importance)),
    avg_rna_importance,
    color=rna_color,
    alpha=0.85
)
axes[1].errorbar(
    range(len(avg_rna_importance)),
    avg_rna_importance,
    yerr=std_rna_importance,
    fmt='none',
    ecolor=err_color,
    alpha=0.3
)

axes[1].set_title(f'Average RNA Feature Importance ({n_splits}-Fold CV)', fontsize=18)
axes[1].set_xlabel('Feature Index (R1 to R256)', fontsize=14)
axes[1].set_ylabel('Average Gradient Magnitude', fontsize=14)
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].set_xlim(-10, len(avg_rna_importance) + 10)


# -------- 保存 --------
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'feature_importance_{data_name}_average_{n_splits}folds.png', dpi=300)
plt.show()

print(f"分析图已保存为: feature_importance_{data_name}_average_{n_splits}folds.png")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 准备数据 (已用您最新的数据替换) ---
data = {
    'Dataset': [
        'RPI488', 'RPI488', 'RPI488', 'RPI488', 'RPI488', 'RPI488',
        'RPI1807', 'RPI1807', 'RPI1807', 'RPI1807', 'RPI1807', 'RPI1807',
        'NPInter2', 'NPInter2', 'NPInter2', 'NPInter2', 'NPInter2', 'NPInter2',
        'ZEA22133', 'ZEA22133', 'ZEA22133', 'ZEA22133', 'ZEA22133', 'ZEA22133'
    ],
    'Model': [
        'Base models', 'w/o LNN', 'w/o Dual-Path', 'w/o AdaAttn', 'w/o Warmup', 'LNN-DACA',
        'Base models', 'w/o LNN', 'w/o Dual-Path', 'w/o AdaAttn', 'w/o Warmup', 'LNN-DACA',
        'Base models', 'w/o LNN', 'w/o Dual-Path', 'w/o AdaAttn', 'w/o Warmup', 'LNN-DACA',
        'Base models', 'w/o LNN', 'w/o Dual-Path', 'w/o AdaAttn', 'w/o Warmup', 'LNN-DACA'
    ],
    'AUC':      [90.7, 94.2, 94.2, 93.1, 95.6, 97.8, 92.5, 98.9, 99.1, 99.1, 99.2, 99.5, 96.5, 97.0, 97.0, 96.9, 97.3, 98.7, 95.6, 97.8, 97.7, 97.8, 98.2, 99.5],
    'SEN':      [93.8, 95.7, 95.3, 93.8, 95.1, 96.3, 98.7, 99.8, 98.4, 98.7, 99.2, 98.9, 97.5, 99.2, 97.9, 97.7, 98.0, 98.3, 94.3, 95.3, 94.5, 94.4, 96.2, 96.7],
    'PRE':      [92.4, 94.4, 93.0, 93.9, 95.1, 97.2, 82.1, 96.1, 96.5, 95.4, 97.1, 97.7, 92.7, 93.5, 93.5, 93.2, 94.1, 94.3, 87.2, 92.0, 92.0, 92.2, 90.2, 96.9],
    'F1-Score': [88.5, 89.8, 89.4, 88.8, 92.1, 96.7, 86.7, 96.6, 94.5, 96.1, 97.5, 98.3, 93.2, 93.5, 93.6, 93.5, 94.5, 96.2, 87.8, 91.7, 91.7, 91.9, 92.3, 96.8],
    'ACC':      [88.9, 90.5, 89.8, 89.4, 91.4, 96.7, 84.4, 96.2, 95.8, 96.2, 97.2, 98.1, 92.9, 93.3, 93.4, 93.4, 94.4, 96.1, 87.6, 91.6, 91.6, 91.9, 92.1, 96.8],
    'SPE':      [92.7, 95.5, 94.3, 96.0, 95.7, 97.0, 75.0, 95.9, 96.2, 96.5, 96.5, 97.1, 93.5, 94.7, 93.6, 94.4, 94.6, 94.1, 87.2, 92.2, 92.1, 92.2, 89.7, 96.9],
    'MCC':      [77.7, 81.0, 79.0, 78.5, 82.7, 93.3, 68.6, 92.4, 91.4, 92.5, 94.5, 96.2, 86.0, 86.7, 86.9, 86.9, 88.9, 92.4, 75.3, 83.3, 83.3, 83.8, 84.3, 93.6]
}
df = pd.DataFrame(data)

# --- 2. 转换数据并设定顺序 ---
df_melted = df.melt(id_vars=['Dataset', 'Model'], var_name='Metric', value_name='Score')
model_order = ['Base models', 'w/o LNN', 'w/o Dual-Path', 'w/o AdaAttn', 'w/o Warmup', 'LNN-DACA']
df_melted['Model'] = pd.Categorical(df_melted['Model'], categories=model_order, ordered=True)

# --- 3. 绘图设置 ---
sns.set_style("whitegrid", {'axes.grid': True})
sns.despine(left=True, bottom=True)

datasets = ['RPI488', 'RPI1807', 'NPInter2', 'ZEA22133']

# --- 4. 绘制 2x2 网格图 ---
# 设置尺寸和高 DPI 保持清晰
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Performance Trend of LNN-DACA in Ablation Experiments', fontsize=22, y=0.98)
axes_flat = axes.flatten()

for i, dataset in enumerate(datasets):
    ax = axes_flat[i]
    subset = df_melted[df_melted['Dataset'] == dataset]

    sns.lineplot(
        data=subset,
        x='Model',
        y='Score',
        hue='Metric',
        style='Metric',
        markers=True,
        dashes=False,
        linewidth=2.5,
        ax=ax
    )

    ax.set_title(f'Dataset: {dataset}', fontsize=16)
    ax.set_xlabel(None)
    ax.set_ylabel('Performance Score (%)', fontsize=12)
    ax.tick_params(axis='x', rotation=30, labelsize=11)

    min_score = df_melted['Score'].min()
    ax.set_ylim(bottom=round(min_score - 10, -1), top=100)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.get_legend().remove()

# --- 创建统一的总图例 ---
handles, labels = axes_flat[0].get_legend_handles_labels()
metric_order = df_melted['Metric'].unique()
ordered_handles = [handles[labels.index(m)] for m in metric_order]
ordered_labels = [m for m in metric_order]

fig.legend(
    ordered_handles, ordered_labels,
    title='Metric',
    loc='lower center',
    bbox_to_anchor=(0.5, 0.01),
    ncol=len(metric_order),
    fontsize=12,
    title_fontsize=14
)

# --- 调整整体布局并保存 ---
plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.2, bottom=0.15, top=0.93)
plt.savefig('ablation_study_line_plot_v2.png', dpi=600)
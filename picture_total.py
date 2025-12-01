import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 准备数据 (与之前相同) ---
data = {
    'Dataset': [
        'RPI488','RPI488','RPI488','RPI488','RPI488','RPI488','RPI488',
        'RPI1807','RPI1807','RPI1807','RPI1807','RPI1807','RPI1807','RPI1807',
        'NPInter2','NPInter2','NPInter2','NPInter2','NPInter2','NPInter2',
        'ZEA22133','ZEA22133','ZEA22133'
    ],
    'Model': [
        'LPIDF','RPI-EDLCN','MHAM','CCGNN','Graph-RPI','SeqMG-RPI','LNN-DACA',
        'LPIDF','RPI-EDLCN','MHAM','CCGNN','Graph-RPI','SeqMG-RPI','LNN-DACA',
        'LPIDF','RPI-EDLCN','MHAM','CCGNN','ZHMolGraph','LNN-DACA',
        'MHAM','SeqMG-RPI','LNN-DACA'
    ],
    'AUC':      [89.9,90.4,91.2,91.0,92.3,89.5,98.4, 97.4,97.6,98.2,98.6,98.6,96.1,99.6, 96.6,97.0,96.9,97.2,98.6,98.7, 89.1,90.2,99.4],
    'SEN':      [79.2,81.6,83.8,82.2,86.4,83.6,95.9, 93.9,95.6,93.8,94.2,98.6,98.7,99.3, 93.8,94.5,94.5,91.2,97.5,99.0, 88.8,90.4,98.1],
    'PRE':      [92.5,91.2,89.7,90.2,89.7,94.8,97.2, 95.0,95.3,95.5,97.0,97.3,95.0,97.2, 91.0,91.1,90.9,92.3,93.5,95.2, 88.4,90.1,95.6],
    'F1-Score': [85.3,86.1,86.6,86.0,87.7,88.8,94.8, 94.4,95.4,94.6,95.6,97.9,96.8,98.1, 92.4,92.8,92.7,91.7,95.5,96.5, 83.4,90.2,95.8],
    'ACC':      [85.9,86.8,87.5,87.8,88.0,89.4,94.9, 93.8,94.6,94.1,94.9,97.9,96.4,97.9, 92.3,92.7,92.5,91.8,95.5,96.4, 85.0,90.3,95.8],
    'SPE':      [92.5,92.3,91.1,92.5,89.7,95.4,97.5, 93.7,93.2,94.5,95.9,97.2,93.5,96.4, 90.9,90.9,90.6,92.4,93.8,95.2, 89.6,90.1,95.7],
    'MCC':      [74.9,74.2,74.7,75.4,76.7,79.3,89.6, 87.5,88.8,88.2,89.6,95.8,92.7,95.8, 84.7,85.4,85.1,83.6,91.1,92.9, 71.3,80.4,91.6]
}
df = pd.DataFrame(data)

# --- 2. 转换数据 (与之前相同) ---
metrics_to_plot = ['ACC', 'SEN', 'SPE', 'PRE', 'F1-Score', 'MCC', 'AUC']
df_melted = df.melt(id_vars=['Dataset', 'Model'], value_vars=metrics_to_plot,
                    var_name='Metric', value_name='Performance')

# --- 3. 绘图设置 (修正 LNN-DACA 颜色) ---
sns.set_style("ticks")
# 确保 LNN-DACA 在 model_order 中，并移除未在数据中的 DaLNPI
model_order = ['LPIDF','RPI-EDLCN','MHAM','CCGNN','Graph-RPI','SeqMG-RPI','ZHMolGraph','LNN-DACA']
base_models = [m for m in model_order if m != 'LNN-DACA']
base_colors = sns.color_palette("viridis", n_colors=len(base_models))

color_palette = {model: color for model, color in zip(base_models, base_colors)}
color_palette['LNN-DACA'] = 'red' # 将 LNN-DACA 设置为红色突出

# --- 4. 绘制 2x2 网格图 ---
# 通过 figsize=(20, 15) 设置图片大小
fig, axes = plt.subplots(2, 2, figsize=(20, 15))
fig.suptitle('Overall Performance Comparison of Models', fontsize=24, y=0.98)

axes_flat = axes.flatten()
datasets = ['RPI488', 'RPI1807', 'NPInter2', 'ZEA22133']

for i, dataset in enumerate(datasets):
    ax = axes_flat[i]
    subset = df_melted[df_melted['Dataset'] == dataset]
    # 过滤出当前数据集实际包含的模型
    current_model_order = [m for m in model_order if m in subset['Model'].unique()]

    sns.barplot(x='Metric', y='Performance', hue='Model', data=subset,
                ax=ax, palette=color_palette, order=metrics_to_plot, hue_order=current_model_order)

    ax.set_title(f'Dataset: {dataset}', fontsize=18)
    ax.set_xlabel(None)
    ax.set_ylabel('Performance (%)', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.set_ylim(70, 102)
    ax.get_legend().remove()

# --- 创建统一的总图例 (与之前相同) ---
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels,
           title='Models',
           loc='lower center',
           bbox_to_anchor=(0.5, 0.01),
           ncol=len(model_order),
           fontsize=12,
           title_fontsize=14)

# --- 修改点：使用 subplots_adjust 手动调整间距 ---
plt.tight_layout()
plt.subplots_adjust(hspace=0.4, bottom=0.15, top=0.92)

# --- 关键修改：使用 plt.savefig() 替换 plt.show() ---
# 通过 dpi=300 参数设置高分辨率和清晰度
plt.savefig('performance_comparison.png', dpi=300)
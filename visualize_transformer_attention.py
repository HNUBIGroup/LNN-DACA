import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from models.lnndaca_transformer import LNNDACA_Transformer
from dataset_seq import RPI1807_SeqDataset


# ============================================================
# 1. 数据加载 + 选择样本
# ============================================================

excel = "data/RPI1807/RPI1807.xlsx"
rna_fa = "data/RPI1807/RPI1807_rna_seq.fa"
pro_fa = "data/RPI1807/RPI1807_protein_seq.fa"

dataset = RPI1807_SeqDataset(excel, rna_fa, pro_fa)

idx = 100
rna_seq, pro_seq, label = dataset[idx]

print("RNA length:", len(rna_seq))
print("Protein length:", len(pro_seq))
print("Label:", label.item())



# ============================================================
# 2. 加载模型（新结构）
# ============================================================

model = LNNDACA_Transformer(
    embed_dim=128,
    heads=8,
    Lr=500,
    Lp=600,
    enc_layers=2
)

ckpt = "best_model_transformer.pt"
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()

print("Model loaded.")



# ============================================================
# 3. 前向推理，提取注意力
# ============================================================

HEAD = 3   # 只在这里定义一次（最重要）

with torch.no_grad():
    logits, (attn, base_logits, tau, rna_feat, pro_feat) = model([rna_seq], [pro_seq])

# 提取单头注意力
attn = attn[0].numpy()               # [H, Lp, Lr]
A_head = attn[HEAD]                  # [Lp, Lr]

# 提取单头 base logits
base_logits = base_logits[0][HEAD].numpy()     # ← 正确写法

print("Adaptive τ =", tau)



# ============================================================
# 4. Gamma 增强
# ============================================================

def gamma_enhance(A, gamma):
    A_min, A_max = A.min(), A.max()
    A_norm = (A - A_min) / (A_max - A_min + 1e-9)
    return A_norm ** gamma



# ============================================================
# 5. 重新计算不同 τ 的注意力（极端增强版）
# ============================================================

def recompute_tau(base_logits, tau):
    Lp, Lr = base_logits.shape
    logits = base_logits / tau
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / (exp.sum(axis=-1, keepdims=True) + 1e-9)

tau_orig = tau
tau_high = tau * 4.0
tau_low  = tau * 0.15

print(f"τ_orig={tau_orig:.3f}, τ×4={tau_high:.3f}, τ×0.15={tau_low:.3f}")

A_orig = gamma_enhance(recompute_tau(base_logits, tau_orig), gamma=0.7)
A_high = gamma_enhance(recompute_tau(base_logits, tau_high), gamma=0.4)
A_low  = gamma_enhance(recompute_tau(base_logits, tau_low),  gamma=1.8)


# 截断到真实长度
Lr_vis = len(rna_seq)
Lp_vis = len(pro_seq)

A_orig = A_orig[:Lp_vis, :Lr_vis]
A_high = A_high[:Lp_vis, :Lr_vis]
A_low  = A_low[:Lp_vis, :Lr_vis]



# ============================================================
# 6. 绘制三联图（A/B/C），论文级样式
# ============================================================

plt.figure(figsize=(21, 6))

plt.subplot(1, 3, 1)
sns.heatmap(A_orig, cmap="viridis")
plt.title(f"A. Original Attention (τ={tau_orig:.3f})", fontsize=18, fontweight="bold")
plt.xlabel("RNA positions")
plt.ylabel("Protein positions")

plt.subplot(1, 3, 2)
sns.heatmap(A_high, cmap="viridis")
plt.title("B. High Temperature (τ × 4)", fontsize=18, fontweight="bold")
plt.xlabel("RNA positions")
plt.ylabel("Protein positions")

plt.subplot(1, 3, 3)
sns.heatmap(A_low, cmap="viridis")
plt.title("C. Low Temperature (τ × 0.15)", fontsize=18, fontweight="bold")
plt.xlabel("RNA positions")
plt.ylabel("Protein positions")

plt.tight_layout()
plt.savefig("attention_final.png", dpi=600, bbox_inches="tight")
plt.show()

print("\nSaved final figure: attention_final.png")

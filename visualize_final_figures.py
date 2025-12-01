import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from models.lnndaca_transformer import LNNDACA_Transformer
from dataset_seq import RPI1807_SeqDataset


# ===========================================
# 0. 全局样式（简约 Nature 风）
# ===========================================
sns.set_theme(style="white")
plt.rcParams.update({
    "font.size": 14,
    "font.family": "Arial",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

# ===========================================
# 1. 数据加载
# ===========================================
excel = "data/RPI1807/RPI1807.xlsx"
rna_fa = "data/RPI1807/RPI1807_rna_seq.fa"
pro_fa = "data/RPI1807/RPI1807_protein_seq.fa"

dataset = RPI1807_SeqDataset(excel, rna_fa, pro_fa)

idx = 100
rna_seq, pro_seq, label = dataset[idx]

print("RNA length:", len(rna_seq))
print("Protein length:", len(pro_seq))

# ===========================================
# 2. 加载模型
# ===========================================
model = LNNDACA_Transformer(embed_dim=128, heads=8, Lr=500, Lp=600, enc_layers=2)
ckpt = "best_model_transformer.pt"

print("Loading checkpoint:", ckpt)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()
print("Model loaded.\n")

# ===========================================
# 3. 前向推理
# ===========================================
with torch.no_grad():
    logits, (attn, base_logits, tau, rna_feat, pro_feat) = model([rna_seq], [pro_seq])

attn = attn.cpu().numpy()               # shape (1, H, Lp, Lr)
base_logits = base_logits.cpu().numpy() # shape (1, H, Lp, Lr)

print("base_logits shape =", base_logits.shape)

# 去掉 batch 维度 → shape (H, Lp, Lr)
base_logits = base_logits[0]
attn = attn[0]

H, Lp, Lr = attn.shape
print(f"Attention heads = {H}")
print(f"Adaptive tau = {tau}\n")

# ===========================================
# 4. Feature similarity map
# ===========================================
rna_f = rna_feat[0].cpu().numpy()      # [Lr, D]
pro_f = pro_feat[0].cpu().numpy()      # [Lp, D]

sim = (rna_f @ pro_f.T)
sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-9)

# ===========================================
# 5. 选择注意力 head
# ===========================================
HEAD = 3 if H > 3 else 0
print(f"Using HEAD = {HEAD}")

base = base_logits[HEAD]  # shape (Lp, Lr)

# ===========================================
# 6. 温度重计算
# ===========================================
def recompute_tau(logits, tau):
    logits = logits / tau
    logits -= logits.max(axis=-1, keepdims=True)
    expv = np.exp(logits)
    return expv / (expv.sum(axis=-1, keepdims=True) + 1e-9)

tau_orig = float(tau)
tau_high = tau_orig * 4.0
tau_low  = tau_orig * 0.15

A_orig = recompute_tau(base, tau_orig)
A_high = recompute_tau(base, tau_high)
A_low  = recompute_tau(base, tau_low)

# 裁剪长度
Lp_vis, Lr_vis = len(pro_seq), len(rna_seq)

A_orig = A_orig[:Lp_vis, :Lr_vis]
A_high = A_high[:Lp_vis, :Lr_vis]
A_low  = A_low[:Lp_vis, :Lr_vis]

# ===========================================
# 7. Top-k map
# ===========================================
K = max(1, int(0.01 * Lp_vis * Lr_vis))
flat = A_orig.flatten()
thr = np.partition(flat, -K)[-K]
topk_map = (A_orig >= thr).astype(float)

# ===========================================
# 8. 保存单图
# ===========================================
def save_heatmap(M, title, outname):
    plt.figure(figsize=(7, 6))
    sns.heatmap(M, cmap="viridis", cbar=True,
                xticklabels=False, yticklabels=False)
    plt.title(title, fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outname, dpi=600)
    plt.close()

save_heatmap(sim, "RNA–Protein Feature Similarity Map", "Fig5A_similarity_map.png")
save_heatmap(A_orig, f"A. Original Attention (τ={tau_orig:.3f})", "Fig5B_attention_original.png")
save_heatmap(A_high, "B. High Temperature (τ × 4)", "Fig5C_attention_highT.png")
save_heatmap(A_low, "C. Low Temperature (τ × 0.15)", "Fig5D_attention_lowT.png")
save_heatmap(topk_map, "D. Top-1% Interaction Map", "Fig5E_topk_interaction.png")

# ===========================================
# 9. 整合大图
# ===========================================
fig, axes = plt.subplots(1, 5, figsize=(28, 6))
titles = ["Feature", "Original", "High Temp", "Low Temp", "Top-k"]
maps = [sim, A_orig, A_high, A_low, topk_map]

for ax, M, t in zip(axes, maps, titles):
    sns.heatmap(M, cmap="viridis", ax=ax, xticklabels=False, yticklabels=False, cbar=False)
    ax.set_title(t, fontsize=16, fontweight="bold")

plt.tight_layout()
plt.savefig("Fig5_all_in_one.png", dpi=600)
plt.close()

print("\n✅ All DONE! Saved:",
      "\n  Fig5A_similarity_map.png",
      "\n  Fig5B_attention_original.png",
      "\n  Fig5C_attention_highT.png",
      "\n  Fig5D_attention_lowT.png",
      "\n  Fig5E_topk_interaction.png",
      "\n  Fig5_all_in_one.png")

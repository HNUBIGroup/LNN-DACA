import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from dataset_seq import RPI1807_SeqDataset
from models.lnndaca_transformer import LNNDACA_Transformer


# ==========================================================
# 1. Device
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==========================================================
# 2. Dataset paths
# ==========================================================

excel_path = "data/RPI1807/RPI1807.xlsx"
rna_fasta = "data/RPI1807/RPI1807_rna_seq.fa"
pro_fasta = "data/RPI1807/RPI1807_protein_seq.fa"

batch_size = 8
epochs = 10
lr = 1e-4


# ==========================================================
# 3. Load dataset
# ==========================================================

print("\nLoading dataset...")
dataset = RPI1807_SeqDataset(excel_path, rna_fasta, pro_fasta)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


# ==========================================================
# 4. Model + Loss + Optimizer
# ==========================================================

model = LNNDACA_Transformer(
    embed_dim=128,
    heads=8,
    Lr=500,
    Lp=600,
    enc_layers=2
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")


# ==========================================================
# 5. Validation function
# ==========================================================

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for rna, pro, labels in loader:
            logits, _ = model(rna, pro)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            lbl = labels.cpu().numpy().flatten()

            y_true.extend(lbl.tolist())
            y_pred.extend(probs.tolist())

    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, [1 if p > 0.5 else 0 for p in y_pred])
    return auc, acc


# ==========================================================
# 6. Training loop
# ==========================================================

best_auc = 0
best_path = "best_model_transformer.pt"

print("\nStart training...\n")

for epoch in range(1, epochs + 1):

    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, total=len(train_loader))

    for rna, pro, labels in loop:
        optimizer.zero_grad()

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(device.type == "cuda")
        ):
            logits, _ = model(rna, pro)
            loss = criterion(logits, labels.to(device))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        loop.set_description(f"Epoch [{epoch}/{epochs}]")
        loop.set_postfix(loss=loss.item())

    # ----- validation -----
    auc, acc = evaluate(model, val_loader)
    print(f"\nEpoch {epoch} | AUC={auc:.4f} | ACC={acc:.4f}\n")

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), best_path)
        print(f"✔ Best model saved to {best_path}")

print("\nTraining completed!")
print(f"Best AUC = {best_auc:.4f}")

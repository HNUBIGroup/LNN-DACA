import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd



# ==========================================================
# 1. FASTA 文件读取
# ==========================================================

def load_fasta(path):
    """
    返回: dict[id] = sequence
    """
    seq_dict = {}
    with open(path, "r") as f:
        curr_id = None
        curr_seq = []

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # 保存上一条序列
                if curr_id is not None:
                    seq_dict[curr_id] = "".join(curr_seq)

                curr_id = line[1:].strip()   # 去掉 >
                curr_seq = []
            else:
                curr_seq.append(line)

        # 保存最后一条
        if curr_id is not None:
            seq_dict[curr_id] = "".join(curr_seq)

    return seq_dict



# ==========================================================
# 2. Dataset 构造函数
# ==========================================================

class RPI1807_SeqDataset(Dataset):
    def __init__(self, excel_path, rna_fasta_path, protein_fasta_path):
        # load excel
        df = pd.read_excel(excel_path)

        # 确保列存在
        assert "RNA names" in df.columns
        assert "Protein names" in df.columns
        assert "Labels" in df.columns

        self.df = df

        # load fasta
        self.rna_dict = load_fasta(rna_fasta_path)
        self.pro_dict = load_fasta(protein_fasta_path)

        # 清洗：只保留 FASTA 中存在 ID 的 pair
        valid_pairs = []
        missing = 0

        for _, row in df.iterrows():
            rna_id = row["RNA names"]
            pro_id = row["Protein names"]
            label = row["Labels"]

            if rna_id in self.rna_dict and pro_id in self.pro_dict:
                valid_pairs.append((rna_id, pro_id, int(label)))
            else:
                missing += 1

        print(f"有效配对数量: {len(valid_pairs)}")
        print(f"找不到序列的 pair 数量: {missing}")

        self.pairs = valid_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rna_id, pro_id, label = self.pairs[idx]

        rna_seq = self.rna_dict[rna_id]
        pro_seq = self.pro_dict[pro_id]

        return rna_seq, pro_seq, torch.tensor([label], dtype=torch.float)



# ==========================================================
# 3. DataLoader 构造函数
# ==========================================================

def get_dataloader(excel_path, rna_fasta, pro_fasta, batch_size=8, shuffle=True):
    dataset = RPI1807_SeqDataset(excel_path, rna_fasta, pro_fasta)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
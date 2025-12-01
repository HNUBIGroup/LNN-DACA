from dataset_seq import RPI1807_SeqDataset

excel = "data/RPI1807/RPI1807.xlsx"
rna = "data/RPI1807/RPI1807_rna_seq.fa"
pro = "data/RPI1807/RPI1807_protein_seq.fa"

dataset = RPI1807_SeqDataset(excel, rna, pro)

max_rna = 0
max_pro = 0

for rna_seq, pro_seq, _ in dataset:
    max_rna = max(max_rna, len(rna_seq))
    max_pro = max(max_pro, len(pro_seq))

print("Max RNA length:", max_rna)
print("Max Protein length:", max_pro)

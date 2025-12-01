import pandas as pd
from collections import Counter

# ============================== 您可以在这里调整“质量”标准 ==============================
# 目标数据集的文件夹名称
DATA_NAME = 'RPI7317'  # <--- 修改这里以处理不同的数据集

# 1. 序列内容标准：只允许这些字符存在
VALID_RNA_CHARS = {'A', 'C', 'G', 'U', 'T'}  # 允许T，代码会自动转为U
VALID_PROTEIN_CHARS = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                       'Y'}

# 2. 序列长度标准：序列长度必须大于或等于这个值
MIN_RNA_LEN = 30  # <--- 推荐从30开始尝试
MIN_PROTEIN_LEN = 30  # <--- 推荐从30开始尝试
# =====================================================================================


# --- 文件路径配置 (通常无需修改) ---
# 原始文件路径
RNA_FASTA_IN = f"data/{DATA_NAME}/{DATA_NAME}_rna_seq.fa"
PROTEIN_FASTA_IN = f"data/{DATA_NAME}/{DATA_NAME}_protein_seq.fa"
INTERACTION_FILE_IN = f"data/{DATA_NAME}/{DATA_NAME}.xlsx"

# 清洗后新文件的保存路径
RNA_FASTA_OUT = f"data/{DATA_NAME}/{DATA_NAME}_rna_seq_cleaned.fa"
PROTEIN_FASTA_OUT = f"data/{DATA_NAME}/{DATA_NAME}_protein_seq_cleaned.fa"
INTERACTION_FILE_OUT = f"data/{DATA_NAME}/{DATA_NAME}_cleaned.xlsx"


# ------------------------------------


def parse_fasta(file_path):
    """解析FASTA文件，返回一个ID到序列的字典。"""
    sequences = {}
    current_seq_id = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                current_seq_id = line[1:].split()[0]
                sequences[current_seq_id] = ""
            else:
                if current_seq_id:
                    sequences[current_seq_id] += line.upper()
    return sequences


def clean_fasta(input_path, output_path, valid_chars, min_len):
    """
    清洗FASTA文件：移除包含无效字符和过短的序列。
    返回一个包含所有“高质量”序列ID的集合。
    """
    print(f"--- 正在清洗文件: {input_path} ---")

    sequences = parse_fasta(input_path)
    valid_seq_ids = set()

    original_count = len(sequences)

    with open(output_path, 'w') as f_out:
        for seq_id, sequence in sequences.items():
            # 检查1: 是否包含无效字符
            if not set(sequence).issubset(valid_chars):
                print(f"  [删除] 序列 '{seq_id}' 包含无效字符。")
                continue

            # 检查2: 序列长度是否达标
            if len(sequence) < min_len:
                print(f"  [删除] 序列 '{seq_id}' 长度 ({len(sequence)}) 过短，未达到阈值 {min_len}。")
                continue

            # 如果通过所有质量检查，则视为高质量序列
            f_out.write(f">{seq_id}\n")
            f_out.write(f"{sequence}\n")
            valid_seq_ids.add(seq_id)

    cleaned_count = len(valid_seq_ids)
    print(
        f"清洗完成。原始数量: {original_count}, 清洗后高质量序列数量: {cleaned_count} ({original_count - cleaned_count}条低质量序列已被删除)")
    print(f"高质量序列文件已保存至: {output_path}\n")
    return valid_seq_ids


def clean_interactions(input_path, output_path, valid_rna_ids, valid_protein_ids):
    """
    根据高质量的RNA和蛋白质ID列表，筛选交互对。
    """
    print(f"--- 正在清洗交互对文件: {input_path} ---")

    df = pd.read_excel(input_path)
    original_count = len(df)

    # 筛选，只保留RNA和蛋白质都属于高质量列表的交互对
    cleaned_df = df[
        df['RNA names'].isin(valid_rna_ids) &
        df['Protein names'].isin(valid_protein_ids)
        ]

    cleaned_count = len(cleaned_df)

    # 保存到新的Excel文件
    cleaned_df.to_excel(output_path, index=False)

    print(
        f"清洗完成。原始交互对: {original_count}, 清洗后高质量交互对: {cleaned_count} ({original_count - cleaned_count}对已被删除)")

    # 打印清洗后的正负样本统计
    if cleaned_count > 0:
        stats = Counter(cleaned_df['label'])
        print(f"清洗后正样本(1)数量: {stats[1]}, 负样本(0)数量: {stats[0]}")

    print(f"高质量交互对文件已保存至: {output_path}\n")


if __name__ == '__main__':
    # 1. 清洗RNA序列，找出所有高质量的RNA
    valid_rna_ids = clean_fasta(RNA_FASTA_IN, RNA_FASTA_OUT, VALID_RNA_CHARS, MIN_RNA_LEN)

    # 2. 清洗蛋白质序列，找出所有高质量的蛋白质
    valid_protein_ids = clean_fasta(PROTEIN_FASTA_IN, PROTEIN_FASTA_OUT, VALID_PROTEIN_CHARS, MIN_PROTEIN_LEN)

    # 3. 根据高质量RNA和蛋白质列表，筛选出高质量的交互对
    clean_interactions(INTERACTION_FILE_IN, INTERACTION_FILE_OUT, valid_rna_ids, valid_protein_ids)

    print("=== 所有数据清洗完成！ ===")
    print("现在，请使用新生成的 `_cleaned` 文件来重新生成k-mer特征并训练模型。")
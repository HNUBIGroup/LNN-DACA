import itertools
from collections import Counter


def parse_fasta(file_path):
    """
    解析FASTA文件。

    Args:
        file_path (str): FASTA文件的路径。

    Returns:
        dict: 一个字典，键是序列名，值是序列字符串。
    """
    sequences = {}
    current_seq_id = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                current_seq_id = line[1:].split()[0]  # 获取'>'后面的ID，并忽略空格后的部分
                sequences[current_seq_id] = ""
            else:
                if current_seq_id:
                    sequences[current_seq_id] += line
    return sequences


def generate_rna_4mer_features(input_fasta, output_file):
    """
    从RNA序列的FASTA文件生成4-mer频率特征文件。

    Args:
        input_fasta (str): 输入的RNA FASTA文件路径。
        output_file (str): 输出的特征文件路径。
    """
    print(f"正在从 {input_fasta} 生成RNA 4-mer特征...")

    # 1. 解析FASTA文件
    rna_sequences = parse_fasta(input_fasta)
    if not rna_sequences:
        print("警告：未在输入文件中找到任何RNA序列。")
        return

    # 2. 生成所有可能的4-mer组合
    bases = ['A', 'C', 'G', 'U']  # RNA使用U（尿嘧啶）而不是T
    all_4mers = [''.join(p) for p in itertools.product(bases, repeat=4)]
    kmer_to_index = {kmer: i for i, kmer in enumerate(all_4mers)}

    # 3. 计算并写入特征
    with open(output_file, 'w') as f_out:
        for seq_id, sequence in rna_sequences.items():
            sequence = sequence.upper().replace('T', 'U')  # 确保序列大写并将T替换为U
            k = 4

            # 初始化一个长度为256的全零向量
            feature_vector = [0.0] * len(all_4mers)

            # 统计k-mer数量
            kmers_in_seq = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

            # 如果序列长度小于k，则无法提取k-mer
            if not kmers_in_seq:
                continue

            kmer_counts = Counter(kmers_in_seq)
            total_kmers = len(kmers_in_seq)

            # 计算频率并填充特征向量
            for kmer, count in kmer_counts.items():
                if kmer in kmer_to_index:
                    index = kmer_to_index[kmer]
                    feature_vector[index] = count / total_kmers

            # 写入文件
            f_out.write(f">{seq_id}\n")
            f_out.write(" ".join(f"{val:.3f}" for val in feature_vector) + "\n")

    print(f"RNA 4-mer特征已成功保存到 {output_file}")


def generate_protein_2mer_features(input_fasta, output_file):
    """
    从蛋白质序列的FASTA文件生成2-mer（二肽）频率特征文件。

    Args:
        input_fasta (str): 输入的蛋白质FASTA文件路径。
        output_file (str): 输出的特征文件路径。
    """
    print(f"正在从 {input_fasta} 生成蛋白质 2-mer特征...")

    # 1. 解析FASTA文件
    protein_sequences = parse_fasta(input_fasta)
    if not protein_sequences:
        print("警告：未在输入文件中找到任何蛋白质序列。")
        return

    # 2. 生成所有可能的2-mer（二肽）组合
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    all_2mers = [''.join(p) for p in itertools.product(amino_acids, repeat=2)]
    kmer_to_index = {kmer: i for i, kmer in enumerate(all_2mers)}

    # 3. 计算并写入特征
    with open(output_file, 'w') as f_out:
        for seq_id, sequence in protein_sequences.items():
            sequence = sequence.upper()  # 确保序列大写
            k = 2

            feature_vector = [0.0] * len(all_2mers)

            kmers_in_seq = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

            if not kmers_in_seq:
                continue

            kmer_counts = Counter(kmers_in_seq)
            total_kmers = len(kmers_in_seq)

            for kmer, count in kmer_counts.items():
                if kmer in kmer_to_index:
                    index = kmer_to_index[kmer]
                    feature_vector[index] = count / total_kmers

            f_out.write(f">{seq_id}\n")
            f_out.write(" ".join(f"{val:.3f}" for val in feature_vector) + "\n")

    print(f"蛋白质 2-mer特征已成功保存到 {output_file}")


# ============================== 使用示例 ==============================
if __name__ == '__main__':
    # 1. 生成RNA 4-mer特征文件
    generate_rna_4mer_features(
        # input_fasta: 这是输入的原始RNA序列文件路径
        input_fasta="D:\\XiaZai\\MHAM-NPI-main\\MHAM-NPI-main\\data\\RPI369\\RPI369_rna_seq.fa",

        # output_file: 这是您想生成的RNA特征文件名
        output_file="data/RPI369/lncRNA_4_mer.txt"
    )

    print("-" * 30)

    # 2. 生成蛋白质 2-mer特征文件
    generate_protein_2mer_features(
        # input_fasta: 这是输入的原始蛋白质序列文件路径
        input_fasta="D:\\XiaZai\\MHAM-NPI-main\\MHAM-NPI-main\\data\\RPI369\\RPI369_protein_seq.fa",

        # output_file: 这是您想生成的蛋白质特征文件名
        output_file="data/RPI369/protein_2_mer.txt"
    )
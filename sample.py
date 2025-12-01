import numpy as np
import pandas as pd


def get_data(data_name):
    rna_mer = {}    # 256
    with open('data/{}/lncRNA_4_mer.txt'.format(data_name), 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                key = line[1:]
                rna_mer[key] = []
            else:
                rna_mer[key].extend([float(x) for x in line.split()])

    pro_mer = {}    # 400
    with open('data/{}/protein_2_mer.txt'.format(data_name), 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                key = line[1:]
                pro_mer[key] = []
            else:
                pro_mer[key].extend([float(x) for x in line.split()])

    interaction = pd.read_excel("data/{}/{}.xlsx".format(data_name, data_name))
    rna = list(set(interaction["RNA names"].tolist()))
    pro = list(set(interaction["Protein names"].tolist()))

    all_data = []
    for i in range(len(interaction)):
        rna_name = interaction["RNA names"][i]
        pro_name = interaction["Protein names"][i]
        all_data.append([interaction["Labels"][i]] + rna_mer[rna_name] + pro_mer[pro_name])
        if data_name == 'RPI369':
            x = np.random.randint(len(rna))
            y = np.random.randint(len(pro))
            all_data.append([0] + rna_mer[rna[x]] + pro_mer[pro[y]])

    pd_data = pd.DataFrame(all_data)
    pd_data.to_csv("data/{}/sample.txt".format(data_name), sep="\t", columns=None, header=None)
    print(pd_data)


data_name = 'RPI369'
get_data(data_name)

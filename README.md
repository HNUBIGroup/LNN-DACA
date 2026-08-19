This repository provides the complete implementation of DaLNPI, a dual-path liquid neural network equipped with adaptive cross-modal attention for RNA–protein interaction (RPI) prediction.
The framework includes feature construction, dataset generation, warm-up pretraining, 10-fold cross-validation, and post-hoc feature attribution.

All experiments reported in the accompanying manuscript can be fully reproduced using the scripts provided in this repository.

1. Overview

RNA–protein interactions are essential for post-transcriptional regulation and numerous cellular processes.
DaLNPI integrates multi-scale protein features and high-dimensional RNA embeddings, and models their relationships through:

Multi-branch hierarchical convolutional networks (HMCN)

Liquid neural layers for dynamic sequence modeling

Adaptive-temperature cross-modal attention

Dual-path gated fusion mechanisms

Transformer-based deep feature refinement

The overall architecture is designed to handle non-Euclidean, high-dimensional, and multi-modal RPI data efficiently and effectively.

2. Data Processing and Feature Construction

RNA sequences are parsed from FASTA files, converted to uppercase U-based sequences, and tokenized into all possible 4-mer substrings.
The relative frequencies of all 256 RNA 4-mers are computed for each sequence.

Protein sequences are processed analogously to extract 400 dipeptide (2-mer) frequencies across 20 amino acids.

Feature extraction can be performed using:

python generate_features.py

Due to the file size, the datasets used in this study are provided separately.

The complete data package can be downloaded from Quark Cloud Drive:

https://pan.quark.cn/s/f0b08808143c

After downloading and extracting the files, place the data folder in the root directory of this repository. Please keep the original directory structure unchanged when running the provided scripts.

3. Construction of the Final Input (sample.txt)

The script combines:

The interaction label

256-dimensional RNA 4-mer features

400-dimensional protein 2-mer features

For certain datasets (e.g., RPI369), additional negative samples are generated through random RNA–protein recombination, following established practices in prior studies.

Generate the training input via:

python sample.py


This will create sample.txt under the corresponding dataset folder.

4. Model Architecture (DaLNPI)

The complete model and its components are implemented in:

model.py

4.1 Warm-Up Pretraining (Optional)

To improve optimization stability, a lightweight warm-up network (DaLNPI_warmup) is first trained on a small subset of the data.
This helps initialize embedding layers and dense feature transformations before training the full model.

Run:

python warmup_train.py


This step produces:

dalnpi_warmup_weights_DATASET.pt

5. Ten-Fold Cross-Validation Training

The main training routine performs rigorous 10-fold cross-validation, including automated threshold selection and model checkpointing.

Run:

python main.py

6. Software and Environment Requirements

Python ≥ 3.9

PyTorch ≥ 1.8

numpy

pandas

scikit-learn

matplotlib

seaborn

Install all dependencies via:

conda env create -f environment.yml
conda activate your_env_name

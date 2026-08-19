# T2b experiment: E3 with a fold-safe lightweight topology view plus
# log-transformed three-hop path count and a Resource Allocation (RA)
# three-hop path score.
# Run the matching independent-semantic warmup script first.
#
# IMPORTANT: every fold builds its bipartite graph from TRAINING POSITIVE EDGES
# only. Validation and test edges are never added to the graph. For positive
# training examples, the target edge is removed while computing topology
# features (leave-one-positive-edge-out) to avoid direct label leakage.
# T1 counts simple length-3 paths r-p1-r1-p and feeds log1p(path_count).
# T2b additionally weights each path by the inverse degree product of its
# intermediate protein and RNA nodes. All conflict-safe V2 logic is retained.

from model import DaLNPI
from utile import (
    calculate_metrics,
    checkpoint_selection_score,
    load_rpi_dataframe,
)

from torch.utils.data import DataLoader, TensorDataset

import os
import pickle
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict


# ============================================================
# 1. Reproducibility
# ============================================================
def set_seed(seed: int = 2022) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


warnings.filterwarnings("ignore")

SCRIPT_VERSION = "T2b-threehop-RA-AdamW-conflict-safe-v2-20260806"

SEED = int(os.environ.get("LNN_SEED", "2022"))
set_seed(SEED)


# ============================================================
# 2. Experiment configuration
# These settings must match warmup_revised.py.
# ============================================================
# Change datasets without editing file paths. In PowerShell, for example:
#   $env:LNN_DATA_NAME="RPI488"
# If the environment variable is absent, NPInter2 is used.
data_name = os.environ.get("LNN_DATA_NAME", "NPInter2")

# Only datasets with non-standard filenames need an entry here.
# All other datasets automatically use sample.txt and <data_name>.xlsx.
dataset_file_config = {
    "NPInter2": {
        "sample_file": "sample_balanced.txt",
        "pair_file": "NPInter2_balanced.xlsx",
        "experiment_suffix": "_balanced",
    },
}

default_file_config = {
    "sample_file": "sample.txt",
    "pair_file": f"{data_name}.xlsx",
    "experiment_suffix": "",
}
current_file_config = dataset_file_config.get(
    data_name,
    default_file_config,
)

default_warmup_experiment_name = (
    f"{data_name.lower()}"
    f"{current_file_config['experiment_suffix']}"
    "_independent_scheduler"
)
default_experiment_name = (
    f"{data_name.lower()}"
    f"{current_file_config['experiment_suffix']}"
    "_independent_topology_t2b_threehop_ra_adamw"
)

experiment_name = os.environ.get(
    "LNN_EXPERIMENT_NAME",
    default_experiment_name,
)
warmup_experiment_name = os.environ.get(
    "LNN_WARMUP_EXPERIMENT_NAME",
    default_warmup_experiment_name,
)

batch_size = 64
embedding_size = 64
learning_rate = 1e-4
backbone_learning_rate = 2e-5
weight_decay = 1e-4
gradient_clip_norm = 1.0
scheduler_factor = 0.5
scheduler_patience = 5
minimum_learning_rate = 1e-6

max_epochs = 200
patience = 20
min_delta = 1e-6
n_splits = 5

num_heads = 4
num_latents = 16
att_layer_num = 2
dropout = 0.1

TOPOLOGY_FEATURE_NAMES = (
    "rna_degree",
    "protein_degree",
    "log1p_rna_degree",
    "log1p_protein_degree",
    "degree_product",
    "rna_to_protein_degree_ratio",
    "protein_to_rna_degree_ratio",
    "rna_two_hop_same_type",
    "protein_two_hop_same_type",
    # T1 feature: use log1p to stabilize the long-tailed path count.
    "log1p_three_hop_path_count",
    # T2b feature: sum inverse degree products over simple r-p1-r1-p paths.
    "three_hop_resource_allocation_score",
    "pair_orphan",
)
topology_feature_dim = len(TOPOLOGY_FEATURE_NAMES)
THREE_HOP_FEATURE_INDEX = TOPOLOGY_FEATURE_NAMES.index(
    "log1p_three_hop_path_count"
)
RA_FEATURE_INDEX = TOPOLOGY_FEATURE_NAMES.index(
    "three_hop_resource_allocation_score"
)
topology_clip_value = float(os.environ.get("LNN_TOPOLOGY_CLIP", "6.0"))

# Some legacy RPI datasets contain the same RNA-protein identifier pair with
# both positive and negative labels. Keep every supervised row so the saved
# warmup folds remain aligned, but determine ambiguity separately inside each
# outer fold using TRAINING LABELS ONLY. Ambiguous training pairs are excluded
# from that fold's positive topology graph by default.
conflicting_pair_policy = os.environ.get(
    "LNN_CONFLICTING_PAIR_POLICY",
    "exclude_from_topology",
).lower()
if conflicting_pair_policy not in {
    "error",
    "exclude_from_topology",
    "keep_positive_edge",
}:
    raise ValueError(
        "LNN_CONFLICTING_PAIR_POLICY must be 'error', "
        "'exclude_from_topology', or 'keep_positive_edge'."
    )

# Use one prespecified threshold for fair fold-to-fold and model-to-model
# comparison. This remains independent of the held-out test labels.
decision_threshold = 0.5
snapshot_top_k = int(os.environ.get("LNN_SNAPSHOT_TOP_K", "1"))
checkpoint_selection = os.environ.get(
    "LNN_CHECKPOINT_SELECTION",
    "balanced",
).lower()
if checkpoint_selection not in {"auc", "composite", "balanced"}:
    raise ValueError(
        "LNN_CHECKPOINT_SELECTION must be 'auc', 'composite', or 'balanced'."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warmup_dir = os.path.join(
    "warmup_checkpoints",
    data_name,
    warmup_experiment_name,
)

full_model_dir = os.path.join(
    "full_model_checkpoints",
    data_name,
    experiment_name,
)

os.makedirs(full_model_dir, exist_ok=True)


# ============================================================
# 3. Load composition features and pair-aligned foundation embeddings
# ============================================================
all_protein_features = [f"P{i}" for i in range(1, 401)]
all_rna_features = [f"R{i}" for i in range(1, 257)]
composition_feature_names = all_rna_features + all_protein_features

rna_fm_feature_names = [f"RF{i}" for i in range(1, 641)]
esm2_feature_names = [f"E{i}" for i in range(1, 481)]
foundation_feature_names = rna_fm_feature_names + esm2_feature_names
feature_names = composition_feature_names + foundation_feature_names

sample_path = os.path.join(
    "data",
    data_name,
    current_file_config["sample_file"],
)
pair_path = os.path.join(
    "data",
    data_name,
    current_file_config["pair_file"],
)
rna_fm_path = os.path.join("data", data_name, "rna_fm_embedding.pkl")
esm2_path = os.path.join("data", data_name, "esm2_embedding.pkl")

print(f"Selected sample file: {sample_path}")
print(f"Selected pair file: {pair_path}")
print(f"Experiment name: {experiment_name}")
print(f"Script version: {SCRIPT_VERSION}")


def _embedding_to_numpy(value, expected_dim: int, item_id: str, source: str):
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    if array.shape[0] != expected_dim:
        raise ValueError(
            f"{source} embedding for {item_id!r} has dimension "
            f"{array.shape[0]}, expected {expected_dim}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{source} embedding for {item_id!r} is non-finite.")
    return array


def _standardize_pair_columns(pair_frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common pair-mapping column names across RPI datasets.

    Examples:
    - ``Labels`` / ``Label`` -> ``label``
    - ``RNA name`` / ``RNA_names`` -> ``RNA names``
    - ``Protein name`` / ``Protein_names`` -> ``Protein names``
    """
    aliases = {
        "label": "label",
        "labels": "label",
        "rna name": "RNA names",
        "rna names": "RNA names",
        "rna id": "RNA names",
        "rna ids": "RNA names",
        "protein name": "Protein names",
        "protein names": "Protein names",
        "protein id": "Protein names",
        "protein ids": "Protein names",
    }

    standardized_names = []
    for column in pair_frame.columns:
        original = str(column).strip()
        normalized = original.lower().replace("_", " ").replace("-", " ")
        normalized = " ".join(normalized.split())
        standardized_names.append(aliases.get(normalized, original))

    if len(set(standardized_names)) != len(standardized_names):
        duplicates = sorted(
            {
                name
                for name in standardized_names
                if standardized_names.count(name) > 1
            }
        )
        raise ValueError(
            "Pair mapping contains duplicate columns after normalization: "
            f"{duplicates}"
        )

    pair_frame = pair_frame.copy()
    pair_frame.columns = standardized_names
    return pair_frame


def load_augmented_dataframe() -> pd.DataFrame:
    data_frame = load_rpi_dataframe(sample_path, composition_feature_names)

    if not os.path.exists(pair_path):
        raise FileNotFoundError(f"Pair mapping file not found: {pair_path}")
    pair_frame = _standardize_pair_columns(pd.read_excel(pair_path))
    required_columns = {"RNA names", "Protein names", "label"}
    missing_columns = required_columns - set(pair_frame.columns)
    if missing_columns:
        raise ValueError(
            f"Pair mapping is missing columns: {sorted(missing_columns)}"
        )
    if len(pair_frame) != len(data_frame):
        raise ValueError(
            f"Row-count mismatch: {os.path.basename(sample_path)}={len(data_frame)}, "
            f"{os.path.basename(pair_path)}={len(pair_frame)}."
        )

    sample_labels = data_frame["label"].to_numpy(dtype=np.int64)
    pair_labels = pd.to_numeric(
        pair_frame["label"], errors="raise"
    ).to_numpy(dtype=np.int64)
    if not np.array_equal(sample_labels, pair_labels):
        mismatch = np.flatnonzero(sample_labels != pair_labels)[:10]
        raise ValueError(
            f"{os.path.basename(sample_path)} and {os.path.basename(pair_path)} "
            "are not row-aligned: label mismatches "
            f"at rows {mismatch.tolist()}."
        )

    for required_path in (rna_fm_path, esm2_path):
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Embedding file not found: {required_path}")

    with open(rna_fm_path, "rb") as file:
        rna_fm_dictionary = pickle.load(file)
    with open(esm2_path, "rb") as file:
        esm2_dictionary = pickle.load(file)

    # Normalize identifiers once. Duplicate/conflict handling is intentionally
    # NOT based on validation/test labels here. Each FoldTopologyBuilder below
    # audits only its own training subset.
    pair_frame = pair_frame.copy()
    pair_frame.loc[:, "RNA names"] = (
        pair_frame["RNA names"].astype(str).str.strip()
    )
    pair_frame.loc[:, "Protein names"] = (
        pair_frame["Protein names"].astype(str).str.strip()
    )
    rna_ids = pair_frame["RNA names"].tolist()
    protein_ids = pair_frame["Protein names"].tolist()
    data_frame.loc[:, "RNA names"] = rna_ids
    data_frame.loc[:, "Protein names"] = protein_ids

    duplicate_mask = pair_frame.duplicated(
        subset=["RNA names", "Protein names"],
        keep=False,
    )
    if bool(duplicate_mask.any()):
        duplicate_rows = int(duplicate_mask.sum())
        duplicate_pairs = int(
            pair_frame.loc[
                duplicate_mask,
                ["RNA names", "Protein names"],
            ].drop_duplicates().shape[0]
        )
        print(
            "Duplicate pair audit | "
            f"pairs={duplicate_pairs}, rows={duplicate_rows}. "
            "Conflicting labels will be checked independently inside each "
            "training fold; validation/test labels are not used for graph "
            "construction."
        )

    missing_rna = sorted({item_id for item_id in rna_ids if item_id not in rna_fm_dictionary})
    missing_protein = sorted(
        {item_id for item_id in protein_ids if item_id not in esm2_dictionary}
    )
    if missing_rna:
        raise KeyError(
            f"RNA-FM embeddings are missing {len(missing_rna)} IDs; "
            f"examples: {missing_rna[:10]}"
        )
    if missing_protein:
        raise KeyError(
            f"ESM-2 embeddings are missing {len(missing_protein)} IDs; "
            f"examples: {missing_protein[:10]}"
        )

    rna_matrix = np.stack(
        [
            _embedding_to_numpy(
                rna_fm_dictionary[item_id], 640, item_id, "RNA-FM"
            )
            for item_id in rna_ids
        ],
        axis=0,
    )
    esm2_matrix = np.stack(
        [
            _embedding_to_numpy(
                esm2_dictionary[item_id], 480, item_id, "ESM-2"
            )
            for item_id in protein_ids
        ],
        axis=0,
    )

    data_frame.loc[:, rna_fm_feature_names] = rna_matrix
    data_frame.loc[:, esm2_feature_names] = esm2_matrix

    augmented_values = data_frame[feature_names].to_numpy(dtype=np.float32)
    if not np.isfinite(augmented_values).all():
        raise ValueError("The augmented input matrix contains NaN or infinity.")

    print(
        f"Loaded augmented inputs: {len(data_frame)} pairs | "
        f"composition={len(composition_feature_names)} | "
        f"RNA-FM={len(rna_fm_feature_names)} | "
        f"ESM-2={len(esm2_feature_names)} | total={len(feature_names)}"
    )
    return data_frame


data = load_augmented_dataframe()
rna_features = all_rna_features
protein_features = all_protein_features

# Dictionary insertion order is the physical input order used by the model.
feat_sizes = {feature_name: 1 for feature_name in feature_names}
dnn_feature_columns = [
    (feature_name, "dense")
    for feature_name in feature_names
]


# ============================================================
# 4. Apply the fold-specific scaler fitted during warmup
# ============================================================
def apply_preprocessing(
    dataframe: pd.DataFrame,
    preprocessing_package: dict,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    saved_feature_names = preprocessing_package["feature_names"]
    saved_composition_names = preprocessing_package["composition_feature_names"]
    scaler = preprocessing_package["scaler"]

    if list(saved_feature_names) != feature_names:
        raise ValueError(
            "Feature order in the warmup preprocessor does not match the "
            "current augmented full-model input order."
        )
    if list(saved_composition_names) != composition_feature_names:
        raise ValueError(
            "Composition-feature order differs between warmup and full model."
        )

    dataframe.loc[:, composition_feature_names] = scaler.transform(
        dataframe[composition_feature_names].to_numpy(dtype=np.float32)
    ).astype(np.float32)

    return dataframe


# ============================================================
# 5. DataLoader
# ============================================================
def build_loader(
    dataframe: pd.DataFrame,
    topology_matrix: np.ndarray,
    batch_size_value: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    labels = (
        dataframe["label"]
        .to_numpy(dtype=np.float32)
        .reshape(-1, 1)
    )
    features = dataframe[feature_names].to_numpy(dtype=np.float32)
    topology_matrix = np.asarray(topology_matrix, dtype=np.float32)

    if topology_matrix.shape != (len(dataframe), topology_feature_dim):
        raise ValueError(
            "Topology matrix must have shape "
            f"({len(dataframe)}, {topology_feature_dim}), got "
            f"{topology_matrix.shape}."
        )
    if not np.isfinite(topology_matrix).all():
        raise ValueError("Topology matrix contains NaN or infinity.")

    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(topology_matrix),
        torch.from_numpy(labels),
    )

    if len(dataset) == 0:
        raise ValueError("Cannot construct a DataLoader from an empty dataset.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=min(batch_size_value, len(dataset)),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


class FoldTopologyBuilder:
    """Build leak-resistant pair features from one fold's positive training graph.

    The graph is bipartite: RNA nodes connect only to protein nodes. Validation
    and test pairs are not used to construct adjacency. For a positive training
    pair, its own edge is removed while its topology features are calculated,
    matching the inference setting where the candidate edge is unknown.
    """

    def __init__(
        self,
        training_frame: pd.DataFrame,
        conflict_policy: str = "exclude_from_topology",
    ):
        required = {"RNA names", "Protein names", "label"}
        missing = required - set(training_frame.columns)
        if missing:
            raise ValueError(
                f"Topology construction is missing columns: {sorted(missing)}"
            )
        if conflict_policy not in {
            "error",
            "exclude_from_topology",
            "keep_positive_edge",
        }:
            raise ValueError(f"Unsupported conflict policy: {conflict_policy!r}")

        self.conflict_policy = conflict_policy
        self.rna_to_proteins = defaultdict(set)
        self.protein_to_rnas = defaultdict(set)
        self.edge_counts = Counter()

        # IMPORTANT: conflicting pairs are detected from the CURRENT FOLD'S
        # TRAINING LABELS ONLY. Validation/test labels never influence graph
        # construction or pair exclusion.
        audit_frame = training_frame[
            ["RNA names", "Protein names", "label"]
        ].copy()
        audit_frame.loc[:, "RNA names"] = (
            audit_frame["RNA names"].astype(str).str.strip()
        )
        audit_frame.loc[:, "Protein names"] = (
            audit_frame["Protein names"].astype(str).str.strip()
        )
        audit_frame.loc[:, "label"] = (
            pd.to_numeric(audit_frame["label"], errors="raise").astype(int)
        )

        pair_label_counts = audit_frame.groupby(
            ["RNA names", "Protein names"],
            sort=False,
        )["label"].nunique()
        self.conflicting_pairs = set(
            pair_label_counts[pair_label_counts > 1].index.tolist()
        )

        pair_keys = list(
            zip(
                audit_frame["RNA names"].tolist(),
                audit_frame["Protein names"].tolist(),
            )
        )
        conflicting_row_mask = np.fromiter(
            (pair_key in self.conflicting_pairs for pair_key in pair_keys),
            dtype=bool,
            count=len(pair_keys),
        )
        self.conflicting_rows = audit_frame.loc[
            conflicting_row_mask
        ].copy()
        self.num_conflicting_pairs = int(len(self.conflicting_pairs))
        self.num_conflicting_rows = int(conflicting_row_mask.sum())

        if self.num_conflicting_pairs and conflict_policy == "error":
            raise ValueError(
                "The current training fold contains duplicated RNA-protein "
                "pairs with conflicting labels."
            )

        positive_mask = audit_frame["label"].eq(1).to_numpy(dtype=bool)
        if conflict_policy == "exclude_from_topology":
            excluded_positive_mask = positive_mask & conflicting_row_mask
        else:
            excluded_positive_mask = np.zeros(
                len(audit_frame),
                dtype=bool,
            )

        self.num_excluded_conflicting_positive_rows = int(
            excluded_positive_mask.sum()
        )
        graph_positive_mask = positive_mask & ~excluded_positive_mask
        positive_frame = audit_frame.loc[
            graph_positive_mask,
            ["RNA names", "Protein names"],
        ]

        for rna_id, protein_id in positive_frame.itertuples(index=False, name=None):
            self.rna_to_proteins[rna_id].add(protein_id)
            self.protein_to_rnas[protein_id].add(rna_id)
            self.edge_counts[(rna_id, protein_id)] += 1

        self.num_positive_rows = int(len(positive_frame))
        self.num_unique_edges = int(len(self.edge_counts))
        self.num_rna_nodes = int(len(self.rna_to_proteins))
        self.num_protein_nodes = int(len(self.protein_to_rnas))

    def _rna_two_hop_count(self, rna_id: str, excluded_protein=None) -> int:
        same_type_neighbors = set()
        for protein_id in self.rna_to_proteins.get(rna_id, ()):
            if protein_id == excluded_protein:
                continue
            same_type_neighbors.update(self.protein_to_rnas.get(protein_id, ()))
        same_type_neighbors.discard(rna_id)
        return len(same_type_neighbors)

    def _protein_two_hop_count(self, protein_id: str, excluded_rna=None) -> int:
        same_type_neighbors = set()
        for rna_id in self.protein_to_rnas.get(protein_id, ()):
            if rna_id == excluded_rna:
                continue
            same_type_neighbors.update(self.rna_to_proteins.get(rna_id, ()))
        same_type_neighbors.discard(protein_id)
        return len(same_type_neighbors)

    def _three_hop_path_features(
        self,
        rna_id: str,
        protein_id: str,
    ) -> tuple[int, float]:
        """Return count and RA score for simple paths r-p1-r1-p.

        The candidate endpoints cannot reappear as intermediate nodes:
        ``p1 != protein_id`` and ``r1 != rna_id``. For every valid path, the
        Resource Allocation contribution is

            1 / ((degree(p1) + 1) * (degree(r1) + 1)).

        Degrees are computed only from the current fold's non-conflicting
        positive training graph. Validation/test edges and all negative edges
        are absent. The ``+1`` smoothing keeps the definition stable and
        prevents a single low-degree intermediate node from dominating.
        """
        rna_neighbors = self.rna_to_proteins.get(rna_id, ())
        target_protein_rnas = self.protein_to_rnas.get(protein_id, ())
        if not rna_neighbors or not target_protein_rnas:
            return 0, 0.0

        path_count = 0
        ra_score = 0.0

        for intermediate_protein in rna_neighbors:
            if intermediate_protein == protein_id:
                continue

            intermediate_rnas = self.protein_to_rnas.get(
                intermediate_protein, ()
            )
            if not intermediate_rnas:
                continue

            common_rnas = intermediate_rnas.intersection(target_protein_rnas)
            if not common_rnas:
                continue

            intermediate_protein_degree = len(intermediate_rnas)
            protein_denominator = float(intermediate_protein_degree + 1)

            for intermediate_rna in common_rnas:
                if intermediate_rna == rna_id:
                    continue

                intermediate_rna_degree = len(
                    self.rna_to_proteins.get(intermediate_rna, ())
                )
                path_count += 1
                ra_score += 1.0 / (
                    protein_denominator
                    * float(intermediate_rna_degree + 1)
                )

        return int(path_count), float(ra_score)

    def transform(
        self,
        frame: pd.DataFrame,
        leave_one_positive_edge_out: bool,
    ) -> np.ndarray:
        rows = []
        for rna_id, protein_id, label in frame[
            ["RNA names", "Protein names", "label"]
        ].itertuples(index=False, name=None):
            rna_id = str(rna_id).strip()
            protein_id = str(protein_id).strip()
            label = int(label)

            # Remove the candidate edge for every positive training example.
            # The adjacency uses unique neighbors, so this also prevents any
            # duplicated copy of the same pair from exposing the target link.
            remove_target_edge = bool(
                leave_one_positive_edge_out
                and label == 1
                and self.edge_counts.get((rna_id, protein_id), 0) > 0
            )

            rna_degree = len(self.rna_to_proteins.get(rna_id, ()))
            protein_degree = len(self.protein_to_rnas.get(protein_id, ()))
            if remove_target_edge:
                rna_degree -= 1
                protein_degree -= 1

            excluded_protein = protein_id if remove_target_edge else None
            excluded_rna = rna_id if remove_target_edge else None
            rna_two_hop = self._rna_two_hop_count(
                rna_id,
                excluded_protein=excluded_protein,
            )
            protein_two_hop = self._protein_two_hop_count(
                protein_id,
                excluded_rna=excluded_rna,
            )
            (
                three_hop_path_count,
                three_hop_ra_score,
            ) = self._three_hop_path_features(
                rna_id,
                protein_id,
            )

            rna_degree = max(int(rna_degree), 0)
            protein_degree = max(int(protein_degree), 0)
            degree_product = float(rna_degree * protein_degree)
            rna_to_protein_ratio = float(
                (rna_degree + 1.0) / (protein_degree + 1.0)
            )
            protein_to_rna_ratio = float(
                (protein_degree + 1.0) / (rna_degree + 1.0)
            )
            pair_orphan = float(rna_degree == 0 or protein_degree == 0)

            rows.append(
                [
                    float(rna_degree),
                    float(protein_degree),
                    float(np.log1p(rna_degree)),
                    float(np.log1p(protein_degree)),
                    degree_product,
                    rna_to_protein_ratio,
                    protein_to_rna_ratio,
                    float(rna_two_hop),
                    float(protein_two_hop),
                    float(np.log1p(three_hop_path_count)),
                    float(three_hop_ra_score),
                    pair_orphan,
                ]
            )

        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.shape != (len(frame), topology_feature_dim):
            raise RuntimeError(
                "Unexpected topology feature shape: "
                f"{matrix.shape}, expected ({len(frame)}, {topology_feature_dim})."
            )
        return matrix


def standardize_fold_topology(
    train_raw: np.ndarray,
    val_raw: np.ndarray,
    test_raw: np.ndarray,
):
    """Fit topology scaling on training candidates only."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw).astype(np.float32)
    val_scaled = scaler.transform(val_raw).astype(np.float32)
    test_scaled = scaler.transform(test_raw).astype(np.float32)

    if topology_clip_value > 0:
        train_scaled = np.clip(
            train_scaled, -topology_clip_value, topology_clip_value
        )
        val_scaled = np.clip(
            val_scaled, -topology_clip_value, topology_clip_value
        )
        test_scaled = np.clip(
            test_scaled, -topology_clip_value, topology_clip_value
        )

    return scaler, train_scaled, val_scaled, test_scaled


@torch.no_grad()
def get_topology_predictions(loader: DataLoader, model: nn.Module):
    model.eval()
    targets = []
    probabilities = []
    for x, topology, y in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        topology = topology.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        logits = model.forward_logits(x, topology_features=topology)
        probability = torch.sigmoid(logits)
        targets.append(y.detach().cpu().numpy().reshape(-1))
        probabilities.append(probability.detach().cpu().numpy().reshape(-1))

    if not targets:
        raise ValueError("Cannot predict from an empty DataLoader.")
    return np.concatenate(targets), np.concatenate(probabilities)


@torch.no_grad()
def get_topology_result(
    loader: DataLoader,
    model: nn.Module,
    threshold: float,
):
    target, probability = get_topology_predictions(loader, model)
    auc = roc_auc_score(target, probability)
    sen, pre, f1, acc, spe, mcc = calculate_metrics(
        target,
        probability,
        threshold=threshold,
    )
    return auc, sen, pre, f1, acc, spe, mcc


# ============================================================
# 6. Safe checkpoint loading
# ============================================================
def load_torch_weights(weight_path: str):
    try:
        return torch.load(
            weight_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        return torch.load(
            weight_path,
            map_location=device,
        )


# ============================================================
# 7. Transfer only the shared backbone from the warmup model
# ============================================================
def load_warmup_backbone(
    model: nn.Module,
    warmup_weight_path: str,
) -> list[str]:
    if not os.path.exists(warmup_weight_path):
        raise FileNotFoundError(
            f"Warmup checkpoint not found: {warmup_weight_path}"
        )

    saved_state_dict = load_torch_weights(warmup_weight_path)

    if (
        isinstance(saved_state_dict, dict)
        and "state_dict" in saved_state_dict
    ):
        saved_state_dict = saved_state_dict["state_dict"]

    model_state_dict = model.state_dict()

    expected_backbone_keys = {
        key
        for key in model_state_dict
        if key.startswith("backbone.")
    }

    compatible_backbone = {
        key: value
        for key, value in saved_state_dict.items()
        if (
            key.startswith("backbone.")
            and key in model_state_dict
            and model_state_dict[key].shape == value.shape
        )
    }

    missing_backbone_keys = sorted(
        expected_backbone_keys - set(compatible_backbone)
    )

    if missing_backbone_keys:
        preview = ", ".join(missing_backbone_keys[:10])
        raise RuntimeError(
            "The warmup checkpoint does not contain a complete compatible "
            f"backbone. Missing {len(missing_backbone_keys)} keys. "
            f"Examples: {preview}"
        )

    model_state_dict.update(compatible_backbone)
    model.load_state_dict(model_state_dict)

    return sorted(compatible_backbone)


# ============================================================
# 8. Full-model AdamW helpers
# ============================================================
def build_full_model_adamw(
    model: nn.Module,
    backbone_lr: float,
    new_module_lr: float,
    weight_decay_value: float,
) -> torch.optim.Optimizer:
    """Build AdamW with separate learning rates and selective weight decay.

    Matrix-like weights receive decoupled weight decay. Biases, normalization
    parameters, liquid time constants, and adaptive-temperature parameters do
    not receive weight decay. This matches the warmup regularization principle
    while preserving the lower backbone learning rate used for fine-tuning.
    """
    grouped_parameters = {
        "backbone_decay": [],
        "backbone_no_decay": [],
        "new_decay": [],
        "new_no_decay": [],
    }

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        lower_name = name.lower()
        no_decay = (
            parameter.ndim < 2
            or lower_name.endswith(".bias")
            or "norm" in lower_name
            or "raw_tau" in lower_name
            or "temperature" in lower_name
        )
        parameter_scope = (
            "backbone" if name.startswith("backbone.") else "new"
        )
        decay_scope = "no_decay" if no_decay else "decay"
        grouped_parameters[f"{parameter_scope}_{decay_scope}"].append(
            parameter
        )

    group_settings = (
        (
            "backbone_decay",
            backbone_lr,
            weight_decay_value,
        ),
        (
            "backbone_no_decay",
            backbone_lr,
            0.0,
        ),
        (
            "new_decay",
            new_module_lr,
            weight_decay_value,
        ),
        (
            "new_no_decay",
            new_module_lr,
            0.0,
        ),
    )

    optimizer_groups = []
    for group_name, group_lr, group_weight_decay in group_settings:
        parameters = grouped_parameters[group_name]
        if not parameters:
            continue
        optimizer_groups.append(
            {
                "name": group_name,
                "params": parameters,
                "lr": group_lr,
                "weight_decay": group_weight_decay,
            }
        )

    if not optimizer_groups:
        raise RuntimeError("No trainable parameters were found for AdamW.")

    optimizer = torch.optim.AdamW(optimizer_groups)

    print("Full-model optimizer: AdamW with selective weight decay")
    for group in optimizer.param_groups:
        parameter_count = sum(
            parameter.numel() for parameter in group["params"]
        )
        print(
            "  Optimizer group | "
            f"name={group['name']}, "
            f"parameters={parameter_count}, "
            f"lr={float(group['lr']):.2e}, "
            f"weight_decay={float(group['weight_decay']):.2e}"
        )

    return optimizer


def get_scope_learning_rates(
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """Return representative backbone and new-module learning rates."""
    group_lrs = {
        str(group.get("name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }

    backbone_lr = group_lrs.get(
        "backbone_decay",
        group_lrs.get("backbone_no_decay"),
    )
    new_module_lr = group_lrs.get(
        "new_decay",
        group_lrs.get("new_no_decay"),
    )

    if backbone_lr is None or new_module_lr is None:
        raise RuntimeError(
            "Could not recover backbone/new-module learning rates from "
            "the AdamW parameter groups."
        )

    return float(backbone_lr), float(new_module_lr)


def format_scope_lr_change(
    previous_lrs: list[float],
    current_lrs: list[float],
    optimizer: torch.optim.Optimizer,
) -> str:
    """Format LR changes using optimizer-group names for clear diagnostics."""
    changes = []
    for group, previous_lr, current_lr in zip(
        optimizer.param_groups,
        previous_lrs,
        current_lrs,
    ):
        if current_lr < previous_lr - 1e-15:
            changes.append(
                f"{group.get('name', 'unnamed')}: "
                f"{previous_lr:.2e}->{current_lr:.2e}"
            )
    return ", ".join(changes)


# ============================================================
# 9. Train the full model fold by fold
# ============================================================
metric_names = [
    "auc",
    "sen",
    "pre",
    "f1",
    "acc",
    "spe",
    "mcc",
]

all_fold_metrics = []

print(f"Device: {device}")
print(f"Dataset: {data_name}")
print(f"Samples: {len(data)}")
print(f"Folds: {n_splits}")
print("===== Start full-model cross-validation =====")


for fold in range(1, n_splits + 1):
    print(f"\n{'=' * 72}")
    print(f"Full-model fold {fold}/{n_splits}")
    print(f"{'=' * 72}")

    fold_seed = SEED + fold
    set_seed(fold_seed)

    split_path = os.path.join(
        warmup_dir,
        f"fold_{fold:02d}_indices.npz",
    )
    preprocess_path = os.path.join(
        warmup_dir,
        f"fold_{fold:02d}_preprocess.pkl",
    )
    warmup_weight_path = os.path.join(
        warmup_dir,
        f"fold_{fold:02d}_warmup.pt",
    )
    full_model_weight_path = os.path.join(
        full_model_dir,
        f"fold_{fold:02d}_best.pt",
    )

    for required_path in (
        split_path,
        preprocess_path,
        warmup_weight_path,
    ):
        if not os.path.exists(required_path):
            raise FileNotFoundError(
                f"Required fold file is missing: {required_path}\n"
                "Run warmup_revised.py before this script."
            )

    # --------------------------------------------------------
    # 8.1 Reuse the exact train/validation/test split
    # --------------------------------------------------------
    with np.load(split_path) as split_package:
        train_idx = split_package["train_idx"]
        val_idx = split_package["val_idx"]
        test_idx = split_package["test_idx"]

    train_df = data.iloc[train_idx].copy()
    val_df = data.iloc[val_idx].copy()
    test_df = data.iloc[test_idx].copy()

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # --------------------------------------------------------
    # 8.2 Build fold-safe lightweight topology features
    # --------------------------------------------------------
    topology_builder = FoldTopologyBuilder(
        train_df,
        conflict_policy=conflicting_pair_policy,
    )

    if topology_builder.num_conflicting_pairs > 0:
        fold_conflict_report_path = os.path.join(
            full_model_dir,
            f"fold_{fold:02d}_training_conflicting_pair_report.csv",
        )
        topology_builder.conflicting_rows.to_csv(
            fold_conflict_report_path,
            index=True,
            index_label="original_dataframe_index",
        )
        print(
            "Training-fold conflicting pair audit | "
            f"pairs={topology_builder.num_conflicting_pairs}, "
            f"rows={topology_builder.num_conflicting_rows}, "
            "excluded positive rows="
            f"{topology_builder.num_excluded_conflicting_positive_rows}, "
            f"policy={conflicting_pair_policy}. "
            f"Report: {fold_conflict_report_path}"
        )

    train_topology_raw = topology_builder.transform(
        train_df,
        leave_one_positive_edge_out=True,
    )
    val_topology_raw = topology_builder.transform(
        val_df,
        leave_one_positive_edge_out=False,
    )
    test_topology_raw = topology_builder.transform(
        test_df,
        leave_one_positive_edge_out=False,
    )
    (
        topology_scaler,
        train_topology,
        val_topology,
        test_topology,
    ) = standardize_fold_topology(
        train_topology_raw,
        val_topology_raw,
        test_topology_raw,
    )

    train_orphan_rate = float(train_topology_raw[:, -1].mean())
    val_orphan_rate = float(val_topology_raw[:, -1].mean())
    test_orphan_rate = float(test_topology_raw[:, -1].mean())

    # Recover the exact integer counts from log1p solely for diagnostics and
    # saved outputs. These values are not fed directly to the model.
    train_three_hop_count = np.rint(
        np.expm1(train_topology_raw[:, THREE_HOP_FEATURE_INDEX])
    ).astype(np.int64)
    val_three_hop_count = np.rint(
        np.expm1(val_topology_raw[:, THREE_HOP_FEATURE_INDEX])
    ).astype(np.int64)
    test_three_hop_count = np.rint(
        np.expm1(test_topology_raw[:, THREE_HOP_FEATURE_INDEX])
    ).astype(np.int64)

    train_three_hop_ra = train_topology_raw[:, RA_FEATURE_INDEX].astype(
        np.float64
    )
    val_three_hop_ra = val_topology_raw[:, RA_FEATURE_INDEX].astype(
        np.float64
    )
    test_three_hop_ra = test_topology_raw[:, RA_FEATURE_INDEX].astype(
        np.float64
    )

    print(
        "Training positive graph | "
        f"positive rows={topology_builder.num_positive_rows}, "
        "excluded conflicting positive rows="
        f"{topology_builder.num_excluded_conflicting_positive_rows}, "
        f"unique edges={topology_builder.num_unique_edges}, "
        f"RNA nodes={topology_builder.num_rna_nodes}, "
        f"protein nodes={topology_builder.num_protein_nodes}"
    )
    print(
        "Pair orphan rates (train/val/test): "
        f"{train_orphan_rate:.4f}/{val_orphan_rate:.4f}/"
        f"{test_orphan_rate:.4f}"
    )
    print(
        "Three-hop path count mean (train/val/test): "
        f"{train_three_hop_count.mean():.4f}/"
        f"{val_three_hop_count.mean():.4f}/"
        f"{test_three_hop_count.mean():.4f}; "
        "nonzero rates: "
        f"{np.mean(train_three_hop_count > 0):.4f}/"
        f"{np.mean(val_three_hop_count > 0):.4f}/"
        f"{np.mean(test_three_hop_count > 0):.4f}"
    )
    print(
        "Three-hop RA score mean (train/val/test): "
        f"{train_three_hop_ra.mean():.8f}/"
        f"{val_three_hop_ra.mean():.8f}/"
        f"{test_three_hop_ra.mean():.8f}; "
        "nonzero rates: "
        f"{np.mean(train_three_hop_ra > 0):.4f}/"
        f"{np.mean(val_three_hop_ra > 0):.4f}/"
        f"{np.mean(test_three_hop_ra > 0):.4f}"
    )

    topology_package_path = os.path.join(
        full_model_dir,
        f"fold_{fold:02d}_topology.pkl",
    )
    with open(topology_package_path, "wb") as file:
        pickle.dump(
            {
                "feature_names": TOPOLOGY_FEATURE_NAMES,
                "scaler": topology_scaler,
                "training_positive_edges": tuple(topology_builder.edge_counts.keys()),
                "num_positive_rows": topology_builder.num_positive_rows,
                "num_conflicting_pairs": topology_builder.num_conflicting_pairs,
                "num_conflicting_rows": topology_builder.num_conflicting_rows,
                "num_excluded_conflicting_positive_rows": (
                    topology_builder.num_excluded_conflicting_positive_rows
                ),
                "conflicting_pair_policy": conflicting_pair_policy,
                "conflict_detection_scope": "current_training_fold_only",
                "num_unique_edges": topology_builder.num_unique_edges,
                "num_rna_nodes": topology_builder.num_rna_nodes,
                "num_protein_nodes": topology_builder.num_protein_nodes,
                "leave_one_positive_edge_out": True,
                "t1_added_feature": "log1p_three_hop_path_count",
                "t2b_added_feature": (
                    "three_hop_resource_allocation_score"
                ),
                "three_hop_path_definition": "simple r-p1-r1-p paths",
                "three_hop_ra_definition": (
                    "sum 1/((degree(p1)+1)*(degree(r1)+1))"
                ),
                "three_hop_target_endpoints_excluded": True,
                "clip_value": topology_clip_value,
            },
            file,
        )

    # --------------------------------------------------------
    # 8.3 Reuse the exact fold-specific preprocessor
    # --------------------------------------------------------
    with open(preprocess_path, "rb") as file:
        preprocessing_package = pickle.load(file)

    saved_data_name = preprocessing_package.get("data_name", data_name)
    saved_sample_file = preprocessing_package.get(
        "sample_file", current_file_config["sample_file"]
    )
    saved_pair_file = preprocessing_package.get(
        "pair_file", current_file_config["pair_file"]
    )
    if saved_data_name != data_name:
        raise ValueError(
            f"Dataset mismatch: warmup={saved_data_name!r}, "
            f"full model={data_name!r}."
        )
    if saved_sample_file != current_file_config["sample_file"]:
        raise ValueError(
            f"Sample-file mismatch: warmup={saved_sample_file!r}, "
            f"full model={current_file_config['sample_file']!r}."
        )
    if saved_pair_file != current_file_config["pair_file"]:
        raise ValueError(
            f"Pair-file mismatch: warmup={saved_pair_file!r}, "
            f"full model={current_file_config['pair_file']!r}."
        )

    saved_semantic_mode = preprocessing_package.get("semantic_fusion_mode")
    if saved_semantic_mode != "independent_semantic_branch":
        raise ValueError(
            "Semantic-fusion mismatch: "
            f"warmup={saved_semantic_mode!r}, expected='independent_semantic_branch'. "
            "Run the matching warmup script first."
        )

    saved_embedding_size = preprocessing_package["embedding_size"]
    if saved_embedding_size != embedding_size:
        raise ValueError(
            "Embedding size mismatch: "
            f"warmup={saved_embedding_size}, full model={embedding_size}"
        )

    saved_batch_size = preprocessing_package.get("batch_size", batch_size)
    saved_learning_rate = preprocessing_package.get(
        "learning_rate",
        learning_rate,
    )
    saved_weight_decay = preprocessing_package.get(
        "weight_decay",
        weight_decay,
    )

    if saved_batch_size != batch_size:
        raise ValueError(
            f"Batch-size mismatch: warmup={saved_batch_size}, "
            f"full model={batch_size}"
        )
    if not np.isclose(saved_learning_rate, learning_rate):
        raise ValueError(
            f"Learning-rate mismatch: warmup={saved_learning_rate}, "
            f"full model={learning_rate}"
        )
    if not np.isclose(saved_weight_decay, weight_decay):
        raise ValueError(
            f"Weight-decay mismatch: warmup={saved_weight_decay}, "
            f"full model={weight_decay}"
        )

    saved_scheduler_factor = preprocessing_package.get(
        "scheduler_factor", scheduler_factor
    )
    saved_scheduler_patience = preprocessing_package.get(
        "scheduler_patience", scheduler_patience
    )
    saved_minimum_lr = preprocessing_package.get(
        "minimum_learning_rate", minimum_learning_rate
    )
    if not np.isclose(saved_scheduler_factor, scheduler_factor):
        raise ValueError(
            f"Scheduler-factor mismatch: warmup={saved_scheduler_factor}, "
            f"full model={scheduler_factor}"
        )
    if int(saved_scheduler_patience) != scheduler_patience:
        raise ValueError(
            f"Scheduler-patience mismatch: warmup={saved_scheduler_patience}, "
            f"full model={scheduler_patience}"
        )
    if not np.isclose(saved_minimum_lr, minimum_learning_rate):
        raise ValueError(
            f"Minimum-LR mismatch: warmup={saved_minimum_lr}, "
            f"full model={minimum_learning_rate}"
        )

    feat_sizes = preprocessing_package["feat_sizes"]
    dnn_feature_columns = preprocessing_package["dnn_feature_columns"]

    train_df = apply_preprocessing(
        train_df,
        preprocessing_package,
    )
    val_df = apply_preprocessing(
        val_df,
        preprocessing_package,
    )
    test_df = apply_preprocessing(
        test_df,
        preprocessing_package,
    )

    # --------------------------------------------------------
    # 8.3 DataLoaders
    # --------------------------------------------------------
    train_loader = build_loader(
        train_df,
        train_topology,
        batch_size_value=batch_size,
        shuffle=True,
        seed=fold_seed,
    )
    val_loader = build_loader(
        val_df,
        val_topology,
        batch_size_value=batch_size,
        shuffle=False,
        seed=fold_seed,
    )
    test_loader = build_loader(
        test_df,
        test_topology,
        batch_size_value=batch_size,
        shuffle=False,
        seed=fold_seed,
    )

    # --------------------------------------------------------
    # 8.4 Initialize the complete model and transfer backbone
    # --------------------------------------------------------
    model = DaLNPI(
        feat_sizes,
        embedding_size,
        dnn_feature_columns,
        att_layer_num=att_layer_num,
        num_heads=num_heads,
        num_latents=num_latents,
        dropout=dropout,
        topology_feature_dim=topology_feature_dim,
    ).to(device)

    loaded_keys = load_warmup_backbone(
        model,
        warmup_weight_path,
    )

    print(
        f"Loaded {len(loaded_keys)} shared backbone tensors "
        "from this fold's warmup model."
    )

    # Match the warmup regularization principle during full-model fine-tuning:
    # AdamW applies decoupled decay only to matrix-like weights, while biases,
    # LayerNorm parameters, liquid time constants, and temperature parameters
    # are excluded from weight decay. Backbone/new-module LRs remain unchanged.
    loss_func = nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = build_full_model_adamw(
        model=model,
        backbone_lr=backbone_learning_rate,
        new_module_lr=learning_rate,
        weight_decay_value=weight_decay,
    )

    # Drive both parameter-group learning rates with the same validation
    # checkpoint-selection score. Their initial ratio is preserved until a
    # group reaches the common minimum learning rate.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=minimum_learning_rate,
    )

    # --------------------------------------------------------
    # 8.5 Select one checkpoint using validation AUC only
    # --------------------------------------------------------
    best_val_auc = -np.inf
    best_selection_score = -np.inf
    best_epoch = -1
    epochs_without_improvement = 0
    scheduler_reductions = 0
    top_snapshots = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0

        for x, topology, y in train_loader:
            x = x.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            topology = topology.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            y = y.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model.forward_logits(
                x,
                topology_features=topology,
            )
            loss = loss_func(logits, y)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss detected in fold {fold}, epoch {epoch}."
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )
            optimizer.step()

            total_loss += float(loss.item())

        average_loss = total_loss / max(len(train_loader), 1)

        model.eval()
        with torch.no_grad():
            (
                val_auc,
                val_sen,
                val_pre,
                val_f1,
                val_acc,
                val_spe,
                val_mcc,
            ) = get_topology_result(
                val_loader,
                model,
                threshold=decision_threshold,
            )

        if checkpoint_selection == "auc":
            selection_score = float(val_auc)
        elif checkpoint_selection == "composite":
            selection_score = checkpoint_selection_score(
                val_auc,
                val_f1,
                val_mcc,
            )
        else:
            # A stronger convergence guard for the tiny validation folds:
            # random early AUC spikes must also have useful classification.
            selection_score = float(
                0.40 * val_auc
                + 0.30 * val_f1
                + 0.30 * (val_mcc + 1.0) / 2.0
            )

        previous_lrs = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        if np.isfinite(selection_score):
            scheduler.step(float(selection_score))
        current_lrs = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        lr_was_reduced = any(
            current < previous - 1e-15
            for previous, current in zip(previous_lrs, current_lrs)
        )
        if lr_was_reduced:
            scheduler_reductions += 1
        backbone_lr, new_module_lr = get_scope_learning_rates(optimizer)

        # Retain the strongest distinct validation composite-score snapshots.
        # Averaging their probabilities reduces checkpoint-selection variance
        # on small validation sets without consulting the held-out test fold.
        score_is_distinct = not any(
            abs(snapshot["selection_score"] - selection_score) <= 1e-12
            for snapshot in top_snapshots
        )
        if np.isfinite(val_auc) and score_is_distinct:
            snapshot_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            top_snapshots.append(
                {
                    "validation_auc": float(val_auc),
                    "selection_score": selection_score,
                    "epoch": epoch,
                    "state_dict": snapshot_state,
                }
            )
            top_snapshots.sort(
                key=lambda snapshot: snapshot["selection_score"],
                reverse=True,
            )
            del top_snapshots[snapshot_top_k:]

        print(
            f"Fold {fold:02d} | "
            f"Epoch {epoch:03d}/{max_epochs} | "
            f"Loss: {average_loss:.6f} | "
            f"Val AUC: {val_auc:.6f} | "
            f"Val F1: {val_f1:.6f} | "
            f"Val MCC: {val_mcc:.6f} | "
            f"Select: {selection_score:.6f} | "
            f"LR(backbone/new): {backbone_lr:.2e}/{new_module_lr:.2e}"
        )

        if lr_was_reduced:
            lr_change_text = format_scope_lr_change(
                previous_lrs,
                current_lrs,
                optimizer,
            )
            print(
                "  * ReduceLROnPlateau lowered full-model LR | "
                f"{lr_change_text}"
            )

        if (
            np.isfinite(selection_score)
            and selection_score > best_selection_score + min_delta
        ):
            best_val_auc = float(val_auc)
            best_selection_score = float(selection_score)
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                full_model_weight_path,
            )

            print(
                "  * Saved a new best full-model checkpoint: "
                f"selection={best_selection_score:.6f}, "
                f"validation AUC={best_val_auc:.6f}"
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"  Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}, "
                f"best selection={best_selection_score:.6f}, "
                f"validation AUC={best_val_auc:.6f}"
            )
            break

    if not os.path.exists(full_model_weight_path):
        raise RuntimeError(
            f"Fold {fold} did not produce a valid full-model checkpoint."
        )

    # --------------------------------------------------------
    # 8.6 Evaluate the held-out test fold exactly once
    # --------------------------------------------------------
    if not top_snapshots:
        raise RuntimeError(f"Fold {fold} did not retain any valid snapshots.")

    snapshot_probabilities = []
    test_target = None
    snapshot_epochs = []
    snapshot_validation_aucs = []
    for rank, snapshot in enumerate(top_snapshots, start=1):
        model.load_state_dict(snapshot["state_dict"])
        model.eval()
        current_target, current_probability = get_topology_predictions(test_loader, model)
        if test_target is None:
            test_target = current_target
        elif not np.array_equal(test_target, current_target):
            raise RuntimeError("Snapshot test targets are inconsistent.")
        snapshot_probabilities.append(current_probability)
        snapshot_epochs.append(snapshot["epoch"])
        snapshot_validation_aucs.append(snapshot["validation_auc"])
        torch.save(
            snapshot["state_dict"],
            os.path.join(
                full_model_dir,
                f"fold_{fold:02d}_snapshot_{rank:02d}.pt",
            ),
        )

    snapshot_probability_matrix = np.stack(snapshot_probabilities, axis=0)
    test_probability = snapshot_probability_matrix.mean(axis=0)
    test_auc = roc_auc_score(test_target, test_probability)
    (
        test_sen,
        test_pre,
        test_f1,
        test_acc,
        test_spe,
        test_mcc,
    ) = calculate_metrics(
        test_target,
        test_probability,
        threshold=decision_threshold,
    )

    fold_result = {
        "fold": fold,
        "best_epoch": best_epoch,
        "validation_auc": best_val_auc,
        "validation_selection_score": best_selection_score,
        "scheduler_reductions": int(scheduler_reductions),
        "final_backbone_lr": float(backbone_lr),
        "final_new_module_lr": float(new_module_lr),
        "full_model_optimizer": "AdamW",
        "selective_weight_decay": True,
        "decision_threshold": float(decision_threshold),
        "snapshot_count": len(top_snapshots),
        "snapshot_epochs": ";".join(map(str, snapshot_epochs)),
        "snapshot_validation_aucs": ";".join(
            f"{value:.8f}" for value in snapshot_validation_aucs
        ),
        "topology_feature_count": topology_feature_dim,
        "training_graph_positive_rows": topology_builder.num_positive_rows,
        "training_graph_conflicting_pairs": (
            topology_builder.num_conflicting_pairs
        ),
        "training_graph_conflicting_rows": (
            topology_builder.num_conflicting_rows
        ),
        "training_graph_excluded_conflicting_positive_rows": (
            topology_builder.num_excluded_conflicting_positive_rows
        ),
        "conflicting_pair_policy": conflicting_pair_policy,
        "conflict_detection_scope": "current_training_fold_only",
        "training_graph_unique_edges": topology_builder.num_unique_edges,
        "training_graph_rna_nodes": topology_builder.num_rna_nodes,
        "training_graph_protein_nodes": topology_builder.num_protein_nodes,
        "train_pair_orphan_rate": train_orphan_rate,
        "validation_pair_orphan_rate": val_orphan_rate,
        "test_pair_orphan_rate": test_orphan_rate,
        "train_three_hop_mean": float(train_three_hop_count.mean()),
        "validation_three_hop_mean": float(val_three_hop_count.mean()),
        "test_three_hop_mean": float(test_three_hop_count.mean()),
        "train_three_hop_nonzero_rate": float(
            np.mean(train_three_hop_count > 0)
        ),
        "validation_three_hop_nonzero_rate": float(
            np.mean(val_three_hop_count > 0)
        ),
        "test_three_hop_nonzero_rate": float(
            np.mean(test_three_hop_count > 0)
        ),
        "train_three_hop_ra_mean": float(train_three_hop_ra.mean()),
        "validation_three_hop_ra_mean": float(val_three_hop_ra.mean()),
        "test_three_hop_ra_mean": float(test_three_hop_ra.mean()),
        "train_three_hop_ra_nonzero_rate": float(
            np.mean(train_three_hop_ra > 0)
        ),
        "validation_three_hop_ra_nonzero_rate": float(
            np.mean(val_three_hop_ra > 0)
        ),
        "test_three_hop_ra_nonzero_rate": float(
            np.mean(test_three_hop_ra > 0)
        ),
        "auc": float(test_auc),
        "sen": float(test_sen),
        "pre": float(test_pre),
        "f1": float(test_f1),
        "acc": float(test_acc),
        "spe": float(test_spe),
        "mcc": float(test_mcc),
    }
    all_fold_metrics.append(fold_result)

    prediction_path = os.path.join(
        full_model_dir,
        f"fold_{fold:02d}_predictions.npz",
    )
    np.savez(
        prediction_path,
        test_idx=test_idx,
        target=test_target,
        probability=test_probability,
        snapshot_probability=snapshot_probability_matrix,
        topology_features=test_topology,
        topology_features_raw=test_topology_raw,
        three_hop_path_count_raw=test_three_hop_count,
        three_hop_path_count_log1p=test_topology_raw[
            :, THREE_HOP_FEATURE_INDEX
        ],
        three_hop_resource_allocation_score=test_three_hop_ra,
        topology_feature_names=np.asarray(TOPOLOGY_FEATURE_NAMES),
        topology_scaler_mean=np.asarray(topology_scaler.mean_, dtype=np.float64),
        topology_scaler_scale=np.asarray(topology_scaler.scale_, dtype=np.float64),
        decision_threshold=np.asarray(decision_threshold, dtype=np.float64),
        seed=np.asarray(SEED, dtype=np.int64),
    )

    print(
        f"\nFold {fold:02d} final test results | "
        f"AUC: {test_auc:.6f}, "
        f"SEN: {test_sen:.6f}, "
        f"PRE: {test_pre:.6f}, "
        f"F1: {test_f1:.6f}, "
        f"ACC: {test_acc:.6f}, "
        f"SPE: {test_spe:.6f}, "
        f"MCC: {test_mcc:.6f}"
        f", threshold: {decision_threshold:.6f}"
    )

    # Release fold-specific GPU memory before the next fold.
    del (
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        test_loader,
        topology_builder,
        topology_scaler,
        train_topology,
        val_topology,
        test_topology,
        train_topology_raw,
        val_topology_raw,
        test_topology_raw,
        train_three_hop_count,
        val_three_hop_count,
        test_three_hop_count,
        train_three_hop_ra,
        val_three_hop_ra,
        test_three_hop_ra,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 10. Summarize independent test-fold results
# ============================================================
print("\n" + "=" * 72)
print("Independent test-fold summary (mean ± standard deviation)")
print("=" * 72)

summary_rows = []

for metric in metric_names:
    values = np.asarray(
        [fold_result[metric] for fold_result in all_fold_metrics],
        dtype=np.float64,
    )

    mean_value = float(np.mean(values))
    std_value = (
        float(np.std(values, ddof=1))
        if len(values) > 1
        else 0.0
    )

    summary_rows.append(
        {
            "metric": metric,
            "mean": mean_value,
            "standard_deviation": std_value,
        }
    )

    print(
        f"{metric.upper():>4}: "
        f"{mean_value:.6f} ± {std_value:.6f}"
    )


# ============================================================
# 11. Save fold-level and summary results
# ============================================================
fold_results_path = os.path.join(
    full_model_dir,
    f"{data_name}_cross_validation_results.csv",
)
summary_path = os.path.join(
    full_model_dir,
    f"{data_name}_cross_validation_summary.csv",
)

pd.DataFrame(all_fold_metrics).to_csv(
    fold_results_path,
    index=False,
)
pd.DataFrame(summary_rows).to_csv(
    summary_path,
    index=False,
)

print(f"\nFold-level results: {fold_results_path}")
print(f"Summary results: {summary_path}")
print("===== Full-model training and evaluation completed =====")

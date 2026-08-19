# Fold-specific warmup training for E3 independent semantic branch.
# Keeps the E1 scheduler settings and all comparison hyperparameters fixed.

from model import DaLNPI_warmup
from utile import get_result, load_rpi_dataframe

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

import os
import pickle
import random
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# 1. Reproducibility
# ============================================================
def set_seed(seed: int = 2022) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Improve reproducibility on CUDA.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


warnings.filterwarnings("ignore")

SEED = int(os.environ.get("LNN_SEED", "2022"))
set_seed(SEED)


# ============================================================
# 2. Experiment configuration
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

default_experiment_name = (
    f"{data_name.lower()}"
    f"{current_file_config['experiment_suffix']}"
    "_independent_scheduler"
)

experiment_name = os.environ.get(
    "LNN_WARMUP_EXPERIMENT_NAME",
    default_experiment_name,
)

# The revised model creates feature tokens, so 4096 is usually too large.
batch_size = 64

# In the revised model, embedding_size is the feature-token dimension.
# It must be divisible by num_heads=4.
embedding_size = 64

learning_rate = float(os.environ.get("LNN_WARMUP_LR", "1e-4"))
weight_decay = float(os.environ.get("LNN_WARMUP_WEIGHT_DECAY", "1e-4"))
gradient_clip_norm = float(os.environ.get("LNN_GRADIENT_CLIP_NORM", "1.0"))
scheduler_factor = float(os.environ.get("LNN_SCHEDULER_FACTOR", "0.5"))
scheduler_patience = int(os.environ.get("LNN_SCHEDULER_PATIENCE", "5"))
minimum_learning_rate = float(os.environ.get("LNN_MINIMUM_LR", "1e-6"))

max_epochs = int(os.environ.get("LNN_WARMUP_MAX_EPOCHS", "200"))
patience = int(os.environ.get("LNN_WARMUP_PATIENCE", "20"))
minimum_epochs = int(os.environ.get("LNN_WARMUP_MIN_EPOCHS", "15"))
min_delta = float(os.environ.get("LNN_MIN_DELTA", "1e-6"))

n_splits = int(os.environ.get("LNN_N_SPLITS", "5"))
# RPI488's original 10% validation set contains only about 39 samples. A
# slightly larger validation subset makes checkpoint selection less noisy while
# still leaving most development samples for training.
default_validation_ratio = 0.15 if data_name.upper() == "RPI488" else 0.10
validation_ratio = float(
    os.environ.get("LNN_VALIDATION_RATIO", str(default_validation_ratio))
)
if not 0.05 <= validation_ratio <= 0.30:
    raise ValueError("LNN_VALIDATION_RATIO must be between 0.05 and 0.30.")

# Improvements aimed at learning a more transferable warmup backbone.
checkpoint_selection = os.environ.get(
    "LNN_WARMUP_CHECKPOINT_SELECTION", "balanced"
).lower()
if checkpoint_selection not in {"auc", "composite", "balanced"}:
    raise ValueError(
        "LNN_WARMUP_CHECKPOINT_SELECTION must be 'auc', 'composite', "
        "or 'balanced'."
    )

use_ema = os.environ.get("LNN_WARMUP_USE_EMA", "1") != "0"
ema_decay = float(os.environ.get("LNN_WARMUP_EMA_DECAY", "0.995"))
if not 0.0 < ema_decay < 1.0:
    raise ValueError("LNN_WARMUP_EMA_DECAY must be between 0 and 1.")

use_class_weight = os.environ.get("LNN_WARMUP_CLASS_WEIGHT", "1") != "0"
label_smoothing = float(os.environ.get("LNN_WARMUP_LABEL_SMOOTHING", "0.0"))
if not 0.0 <= label_smoothing < 0.5:
    raise ValueError("LNN_WARMUP_LABEL_SMOOTHING must be in [0, 0.5).")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = os.path.join(
    "warmup_checkpoints",
    data_name,
    experiment_name,
)
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 3. Warmup optimization helpers
# ============================================================
def checkpoint_score(
    auc: float,
    f1: float,
    mcc: float,
) -> float:
    """Score checkpoints without consulting the held-out test fold."""
    if checkpoint_selection == "auc":
        return float(auc)
    if checkpoint_selection == "composite":
        # Same equal-weight idea used by many multi-metric selectors.
        return float((auc + f1 + (mcc + 1.0) / 2.0) / 3.0)
    # Balanced default: AUC remains primary, while F1 and MCC prevent a random
    # ranking spike from selecting a practically useless classifier.
    return float(0.40 * auc + 0.30 * f1 + 0.30 * (mcc + 1.0) / 2.0)


class ExponentialMovingAverage:
    """Maintain an EMA copy of every model state tensor.

    EMA usually gives a smoother and more transferable backbone than a single
    noisy optimization step, especially when validation folds are small.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current_state = model.state_dict()
        for name, current in current_state.items():
            shadow = self.shadow[name]
            if torch.is_floating_point(current):
                shadow.mul_(self.decay).add_(
                    current.detach(), alpha=1.0 - self.decay
                )
            else:
                shadow.copy_(current.detach())

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        self.backup = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self.backup:
            raise RuntimeError("EMA restore called before apply_to.")
        model.load_state_dict(self.backup, strict=True)
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in self.shadow.items()
        }


def build_adamw(model: nn.Module) -> torch.optim.Optimizer:
    """Use decoupled weight decay without shrinking biases or normalization."""
    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lower_name = name.lower()
        no_decay = (
            parameter.ndim < 2
            or lower_name.endswith(".bias")
            or "norm" in lower_name
            or "raw_tau" in lower_name
            or "temperature_bias" in lower_name
        )
        if no_decay:
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def smoothed_binary_targets(targets: torch.Tensor) -> torch.Tensor:
    if label_smoothing <= 0.0:
        return targets
    return targets * (1.0 - 2.0 * label_smoothing) + label_smoothing


# ============================================================
# 4. Load composition features and pair-aligned foundation embeddings
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

    rna_ids = pair_frame["RNA names"].astype(str).str.strip().tolist()
    protein_ids = pair_frame["Protein names"].astype(str).str.strip().tolist()

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
# 5. DataLoader
# ============================================================
def build_loader(
    dataframe: pd.DataFrame,
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

    dataset = TensorDataset(
        torch.from_numpy(features),
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


# ============================================================
# 6. Outer stratified cross-validation
# ============================================================
outer_cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=SEED,
)

labels = data["label"].to_numpy(dtype=np.int64)

print(f"Device: {device}")
print(f"Dataset: {data_name}")
print(f"Samples: {len(data)}")
print(f"Positive samples: {int(labels.sum())}")
print(f"Negative samples: {int((labels == 0).sum())}")
print(f"Outer folds: {n_splits}")
print(f"Validation ratio: {validation_ratio:.2f}")
print(f"Checkpoint selection: {checkpoint_selection}")
print(f"EMA enabled: {use_ema} (decay={ema_decay})")
print(f"Class-weighted BCE: {use_class_weight}")
print(f"Label smoothing: {label_smoothing}")
print("===== Start improved fold-specific warmup training =====")


for fold, (development_idx, test_idx) in enumerate(
    outer_cv.split(np.zeros(len(data)), labels),
    start=1,
):
    print(f"\n{'=' * 70}")
    print(f"Warmup fold {fold}/{n_splits}")
    print(f"{'=' * 70}")

    fold_seed = SEED + fold
    set_seed(fold_seed)

    # Split the outer development set into fold-specific training and validation.
    development_labels = labels[development_idx]

    train_idx, val_idx = train_test_split(
        development_idx,
        test_size=validation_ratio,
        random_state=fold_seed,
        shuffle=True,
        stratify=development_labels,
    )

    train_df = data.iloc[train_idx].copy()
    val_df = data.iloc[val_idx].copy()

    print(f"Warmup training samples: {len(train_df)}")
    print(f"Warmup validation samples: {len(val_df)}")
    print(f"Held-out test samples: {len(test_idx)}")

    # --------------------------------------------------------
    # 5.1 Fit one scaler using only this fold's training subset
    # --------------------------------------------------------
    scaler = MinMaxScaler(
        feature_range=(0, 1),
        clip=True,
    )

    # Fit preprocessing only to handcrafted composition features. The frozen
    # foundation-model embeddings retain their pretrained numerical geometry.
    train_df.loc[:, composition_feature_names] = scaler.fit_transform(
        train_df[composition_feature_names]
    ).astype(np.float32)

    val_df.loc[:, composition_feature_names] = scaler.transform(
        val_df[composition_feature_names]
    ).astype(np.float32)

    # --------------------------------------------------------
    # 5.2 Save fold indices and preprocessing state
    # The complete-model script must reuse these exact objects.
    # --------------------------------------------------------
    split_path = os.path.join(
        output_dir,
        f"fold_{fold:02d}_indices.npz",
    )
    np.savez(
        split_path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    preprocess_path = os.path.join(
        output_dir,
        f"fold_{fold:02d}_preprocess.pkl",
    )
    preprocessing_package = {
        "scaler": scaler,
        "feature_names": feature_names,
        "composition_feature_names": composition_feature_names,
        "foundation_feature_names": foundation_feature_names,
        "rna_fm_feature_names": rna_fm_feature_names,
        "esm2_feature_names": esm2_feature_names,
        "protein_features": protein_features,
        "rna_features": rna_features,
        "rna_fm_dim": len(rna_fm_feature_names),
        "esm2_dim": len(esm2_feature_names),
        "feat_sizes": feat_sizes,
        "dnn_feature_columns": dnn_feature_columns,
        "embedding_size": embedding_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "scheduler_name": "ReduceLROnPlateau",
        "scheduler_mode": "max",
        "scheduler_factor": scheduler_factor,
        "scheduler_patience": scheduler_patience,
        "minimum_learning_rate": minimum_learning_rate,
        "random_seed": fold_seed,
        "experiment_name": experiment_name,
        "data_name": data_name,
        "sample_file": current_file_config["sample_file"],
        "pair_file": current_file_config["pair_file"],
        "semantic_fusion_mode": "independent_semantic_branch",
        "warmup_optimizer": "AdamW",
        "warmup_checkpoint_selection": checkpoint_selection,
        "warmup_use_ema": use_ema,
        "warmup_ema_decay": ema_decay,
        "warmup_use_class_weight": use_class_weight,
        "warmup_label_smoothing": label_smoothing,
        "warmup_validation_ratio": validation_ratio,
        "warmup_minimum_epochs": minimum_epochs,
    }

    with open(preprocess_path, "wb") as file:
        pickle.dump(preprocessing_package, file)

    # --------------------------------------------------------
    # 5.3 DataLoaders
    # --------------------------------------------------------
    train_loader = build_loader(
        train_df,
        batch_size_value=batch_size,
        shuffle=True,
        seed=fold_seed,
    )
    val_loader = build_loader(
        val_df,
        batch_size_value=batch_size,
        shuffle=False,
        seed=fold_seed,
    )

    # --------------------------------------------------------
    # 5.4 Warmup model
    # --------------------------------------------------------
    model = DaLNPI_warmup(
        feat_sizes,
        embedding_size,
        dnn_feature_columns,
        att_layer_num=2,
        num_heads=4,
        num_latents=16,
        dropout=0.1,
    ).to(device)

    # Class weighting is neutral (=1) on balanced datasets, but protects the
    # warmup representation when another dataset has an unequal label ratio.
    train_labels = train_df["label"].to_numpy(dtype=np.int64)
    positive_count = int((train_labels == 1).sum())
    negative_count = int((train_labels == 0).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            f"Fold {fold} training subset must contain both classes."
        )
    raw_pos_weight = negative_count / positive_count
    bounded_pos_weight = float(np.clip(raw_pos_weight, 0.25, 4.0))
    pos_weight_tensor = torch.tensor(
        [bounded_pos_weight if use_class_weight else 1.0],
        device=device,
        dtype=torch.float32,
    )

    # Train from logits for numerical stability. AdamW applies decoupled weight
    # decay and excludes biases, normalization parameters, and liquid time
    # constants from shrinkage.
    loss_func = nn.BCEWithLogitsLoss(
        reduction="mean",
        pos_weight=pos_weight_tensor,
    )
    optimizer = build_adamw(model)
    ema = ExponentialMovingAverage(model, ema_decay) if use_ema else None

    print(
        f"Training label counts: positive={positive_count}, "
        f"negative={negative_count}, pos_weight={pos_weight_tensor.item():.4f}"
    )

    # Reduce the learning rate when validation AUC stops improving.
    # The scheduler changes the rate used by the next epoch; it never
    # consults the held-out test fold.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=minimum_learning_rate,
    )

    weight_path = os.path.join(
        output_dir,
        f"fold_{fold:02d}_warmup.pt",
    )

    best_val_auc = -np.inf
    best_val_f1 = -np.inf
    best_val_mcc = -np.inf
    best_selection_score = -np.inf
    best_epoch = -1
    epochs_without_improvement = 0
    scheduler_reductions = 0
    history_rows = []

    # --------------------------------------------------------
    # 5.5 Train and select one checkpoint by validation AUC
    # --------------------------------------------------------
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(
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

            logits = model.forward_logits(x)
            training_targets = smoothed_binary_targets(y)
            loss = loss_func(logits, training_targets)

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
            if ema is not None:
                ema.update(model)

            total_loss += float(loss.item())

        average_loss = total_loss / max(len(train_loader), 1)

        # Validate the EMA-smoothed network when enabled. The raw training
        # weights are restored immediately afterwards and continue optimizing.
        if ema is not None:
            ema.apply_to(model)
        model.eval()
        with torch.no_grad():
            auc, sen, pre, f1, acc, spe, mcc = get_result(
                val_loader,
                model,
            )
        if ema is not None:
            ema.restore(model)

        selection_score = checkpoint_score(auc, f1, mcc)

        previous_lr = float(optimizer.param_groups[0]["lr"])
        if np.isfinite(selection_score):
            scheduler.step(float(selection_score))
        current_lr = float(optimizer.param_groups[0]["lr"])
        lr_was_reduced = current_lr < previous_lr - 1e-15
        if lr_was_reduced:
            scheduler_reductions += 1

        print(
            f"Fold {fold:02d} | "
            f"Epoch {epoch:03d}/{max_epochs} | "
            f"Loss: {average_loss:.6f} | "
            f"Val AUC: {auc:.6f} | "
            f"Val F1: {f1:.6f} | "
            f"Val MCC: {mcc:.6f} | "
            f"Select: {selection_score:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        history_rows.append(
            {
                "epoch": epoch,
                "training_loss": average_loss,
                "validation_auc": float(auc),
                "validation_sen": float(sen),
                "validation_pre": float(pre),
                "validation_f1": float(f1),
                "validation_acc": float(acc),
                "validation_spe": float(spe),
                "validation_mcc": float(mcc),
                "selection_score": float(selection_score),
                "learning_rate": current_lr,
            }
        )

        if lr_was_reduced:
            print(
                "  * ReduceLROnPlateau lowered warmup LR: "
                f"{previous_lr:.2e} -> {current_lr:.2e}"
            )

        score_improved = (
            np.isfinite(selection_score)
            and selection_score > best_selection_score + min_delta
        )
        score_tied_auc_improved = (
            np.isfinite(selection_score)
            and abs(selection_score - best_selection_score) <= min_delta
            and auc > best_val_auc + min_delta
        )

        if score_improved or score_tied_auc_improved:
            best_val_auc = float(auc)
            best_val_f1 = float(f1)
            best_val_mcc = float(mcc)
            best_selection_score = float(selection_score)
            best_epoch = epoch
            epochs_without_improvement = 0

            # Save a pure state_dict for direct backbone transfer. When EMA is
            # enabled, the saved checkpoint is the smoothed model actually used
            # for validation and selection.
            state_to_save = (
                ema.state_dict()
                if ema is not None
                else {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            )
            torch.save(state_to_save, weight_path)

            print(
                "  * Saved a new best warmup checkpoint: "
                f"selection={best_selection_score:.6f}, "
                f"AUC={best_val_auc:.6f}, "
                f"F1={best_val_f1:.6f}, MCC={best_val_mcc:.6f}"
            )
        else:
            epochs_without_improvement += 1

        if epoch >= minimum_epochs and epochs_without_improvement >= patience:
            print(
                f"  Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}, "
                f"best selection={best_selection_score:.6f}, "
                f"validation AUC={best_val_auc:.6f}"
            )
            break

    if not os.path.exists(weight_path):
        raise RuntimeError(
            f"Fold {fold} did not produce a valid warmup checkpoint. "
            "Check the validation labels and metric implementation."
        )

    history_path = os.path.join(
        output_dir,
        f"fold_{fold:02d}_warmup_history.csv",
    )
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    print(
        f"Fold {fold:02d} warmup completed | "
        f"best epoch={best_epoch} | "
        f"best selection={best_selection_score:.6f} | "
        f"AUC={best_val_auc:.6f} | "
        f"F1={best_val_f1:.6f} | MCC={best_val_mcc:.6f} | "
        f"LR reductions={scheduler_reductions} | "
        f"final LR={optimizer.param_groups[0]['lr']:.2e}"
    )
    print(f"Weights: {weight_path}")
    print(f"Preprocessor: {preprocess_path}")
    print(f"Indices: {split_path}")
    print(f"History: {history_path}")

    del model, optimizer, scheduler, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


print("\n===== All improved fold-specific warmup models have been trained =====")

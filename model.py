"""
Self-contained E3 LNN-DACA model with a fold-safe lightweight topology view.

This variant retains the independent RNA-FM/ESM-2 semantic backbone, LNN,
adaptive-temperature bidirectional cross-attention, liquid depth updates, and
adds a sixth topology view projected from fold-specific bipartite-graph features.

This file contains the complete model implementation and does not import another
local model module. It includes:

1. Continuous tokenization of 256 RNA and 400 protein composition features.
2. Learnable latent-token pooling.
3. Dual-path feature processing with a Liquid Neural Network (LNN) branch.
4. Independent RNA-FM (640 -> 128 -> token_dim) and ESM-2
   (480 -> 128 -> token_dim) semantic encoders.
5. Pair-level semantic interaction using [protein, RNA, product, abs difference].
6. Token-wise gated residual semantic injection.
7. Bidirectional sample- and head-specific adaptive-temperature cross-attention.
8. Direction-specific cross-modal gates.
9. Liquid hidden-state updates across stacked interaction layers.
10. Fold-safe lightweight topology-feature projection as a sixth view.
11. Six-view Transformer fusion and final interaction prediction.
12. Post-hoc access to pooling attention, cross-attention, temperatures,
    semantic gates, and intermediate view vectors.

Expected feature names
----------------------
- RNA composition: R1 ... R256
- Protein composition: P1 ... P400
- RNA-FM embedding: RF1 ... RF640
- ESM-2 embedding: E1 ... E480

The physical input-column order may vary because indices are resolved from
feature names. ``embedding_size`` is the token dimension and must be divisible
by ``num_heads``; 64 with 4 heads is the recommended default.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def _numeric_suffix(name: str) -> int:
    """Return the numeric suffix of a feature name such as P17 or R256."""
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 0


def _ordered_feature_names(
    feat_size: Mapping[str, int],
    dnn_feature_columns: Sequence[Tuple[str, str]],
) -> List[str]:
    """
    Recover the order used in the input matrix.

    Dictionary insertion order mirrors the physical matrix column order.  The
    feature-column list is used as a fallback.
    """
    if feat_size:
        return list(feat_size.keys())
    return [name for name, _ in dnn_feature_columns]


# -----------------------------------------------------------------------------
# Continuous feature tokenization and pooling
# -----------------------------------------------------------------------------
class ContinuousFeatureTokenizer(nn.Module):
    """Convert N continuous scalar features into N learnable feature tokens."""

    def __init__(self, num_features: int, token_dim: int, dropout: float = 0.1):
        super().__init__()
        self.num_features = int(num_features)
        self.token_dim = int(token_dim)

        # Shared value projection preserves numeric magnitude relationships.
        self.value_projection = nn.Linear(1, token_dim)

        # Feature identity embedding distinguishes, for example, P1 from P2.
        self.feature_identity = nn.Parameter(
            torch.empty(1, self.num_features, self.token_dim)
        )
        nn.init.normal_(self.feature_identity, mean=0.0, std=0.02)

        self.norm = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected input shape [B, {self.num_features}], got {tuple(x.shape)}"
            )

        tokens = self.value_projection(x.unsqueeze(-1))
        tokens = tokens + self.feature_identity
        return self.dropout(self.norm(tokens))


class LatentTokenPooler(nn.Module):
    """
    Pool a large set of feature tokens into a smaller learnable latent-token set.

    The attention map has shape [B, heads, num_latents, num_input_features], which
    can later be used to relate latent representations back to 4-mer/dipeptide
    feature categories.
    """

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        num_latents: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")

        self.num_latents = int(num_latents)
        self.latent_queries = nn.Parameter(
            torch.empty(1, self.num_latents, token_dim)
        )
        nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)

        self.attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.norm2 = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        queries = self.latent_queries.expand(batch_size, -1, -1)

        pooled, weights = self.attention(
            queries,
            tokens,
            tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attention = weights.detach()

        x = self.norm1(queries + self.dropout(pooled))
        return self.norm2(x + self.dropout(self.ffn(x)))


# -----------------------------------------------------------------------------
# Liquid feature modeling
# -----------------------------------------------------------------------------
class LiquidNeuralLayer1D(nn.Module):
    """
    Gated liquid-state update across a short latent-token sequence.

    This module models dependencies among learned feature tokens. It should not be
    interpreted as processing raw residue or nucleotide positions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        tau: float = 0.15,
        gamma: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.gamma = float(gamma)

        tau = min(max(float(tau), 1e-4), 1.0 - 1e-4)
        self.raw_tau = nn.Parameter(torch.tensor(math.log(tau / (1.0 - tau))))

        self.input_projection = nn.Linear(input_dim, output_dim)
        self.hidden_projection = nn.Linear(output_dim, output_dim, bias=False)
        self.gate = nn.Linear(input_dim + output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, L, C], got {tuple(x.shape)}")

        batch_size, length, _ = x.shape
        hidden = x.new_zeros(batch_size, self.output_dim)
        tau = torch.sigmoid(self.raw_tau)
        outputs: List[torch.Tensor] = []

        for step in range(length):
            x_t = x[:, step, :]
            candidate = torch.tanh(
                self.input_projection(x_t) + self.hidden_projection(hidden)
            )
            update_gate = torch.sigmoid(self.gate(torch.cat([x_t, hidden], dim=-1)))

            leaky_state = (1.0 - tau) * hidden + tau * candidate
            hidden = (
                update_gate * leaky_state
                + (1.0 - update_gate) * (self.gamma * hidden)
            )
            outputs.append(self.dropout(hidden))

        return torch.stack(outputs, dim=1)


class DualPathDynamicFeatureProcessor(nn.Module):
    """Fuse a token-wise nonlinear path with a liquid global-dependency path."""

    def __init__(self, token_dim: int, dropout: float = 0.1):
        super().__init__()
        self.local_path = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.global_path = LiquidNeuralLayer1D(
            token_dim,
            token_dim,
            dropout=dropout,
        )
        self.gate = nn.Linear(token_dim * 2, token_dim)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        local_features = self.local_path(tokens)
        global_features = self.global_path(tokens)
        fusion_gate = torch.sigmoid(
            self.gate(torch.cat([local_features, global_features], dim=-1))
        )
        fused = fusion_gate * global_features + (1.0 - fusion_gate) * local_features
        return self.norm(tokens + fused)


class RNAFeatureEncoder(nn.Module):
    """Lightweight self-attention refinement for pooled RNA feature tokens."""

    def __init__(self, token_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            token_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.norm2 = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        x = self.norm1(tokens + self.dropout(attended))
        return self.norm2(x + self.dropout(self.ffn(x)))


# -----------------------------------------------------------------------------
# Adaptive-temperature cross-modal interaction
# -----------------------------------------------------------------------------
class AdaptiveTemperatureCrossAttention(nn.Module):
    """Multi-head cross-attention with sample- and head-specific temperature.

    Besides performing the original attention calculation, this implementation
    stores the tensors needed for post-hoc interpretability analysis. The added
    attributes are ordinary Python attributes rather than Parameters or persistent
    buffers, so existing checkpoints remain fully compatible.
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        min_temp_factor: float = 0.5,
        max_temp_factor: float = 2.0,
    ):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        if not 0 < min_temp_factor < max_temp_factor:
            raise ValueError("Temperature factors must satisfy 0 < min < max")

        self.feature_dim = int(feature_dim)
        self.num_heads = int(num_heads)
        self.head_dim = feature_dim // num_heads
        self.min_temp_factor = float(min_temp_factor)
        self.max_temp_factor = float(max_temp_factor)

        self.q_projection = nn.Linear(feature_dim, feature_dim)
        self.k_projection = nn.Linear(feature_dim, feature_dim)
        self.v_projection = nn.Linear(feature_dim, feature_dim)
        self.output_projection = nn.Linear(feature_dim, feature_dim)

        # Positive monotonic scale and head-specific bias.
        self.raw_temperature_scale = nn.Parameter(torch.tensor(1.0))
        self.temperature_bias = nn.Parameter(torch.zeros(num_heads))
        self.attention_dropout = nn.Dropout(dropout)

        # Non-persistent interpretability state from the most recent forward pass.
        self.last_attention: Optional[torch.Tensor] = None
        self.last_attention_logits: Optional[torch.Tensor] = None
        self.last_raw_scores: Optional[torch.Tensor] = None
        self.last_similarity: Optional[torch.Tensor] = None
        self.last_temperature_factor: Optional[torch.Tensor] = None
        self.last_learned_temperature: Optional[torch.Tensor] = None
        self.last_temperature: Optional[torch.Tensor] = None

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = tensor.shape
        return (
            tensor.view(batch_size, length, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    @staticmethod
    def _prepare_temperature_override(
        temperature_override,
        learned_temperature: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast an optional positive temperature to [B, H, 1, 1].

        Accepted forms include a scalar, [H], [B, H], or a tensor already
        broadcastable to [B, H, 1, 1]. The override denotes the *actual*
        denominator used in attention, including the sqrt(head_dim) scaling.
        """
        if temperature_override is None:
            return learned_temperature

        override = torch.as_tensor(
            temperature_override,
            device=learned_temperature.device,
            dtype=learned_temperature.dtype,
        )

        if override.ndim == 0:
            override = override.reshape(1, 1, 1, 1)
        elif override.ndim == 1:
            if override.shape[0] != learned_temperature.shape[1]:
                raise ValueError(
                    "A one-dimensional temperature override must have one "
                    "value per attention head."
                )
            override = override.reshape(1, -1, 1, 1)
        elif override.ndim == 2:
            if override.shape != learned_temperature.shape[:2]:
                raise ValueError(
                    "A two-dimensional temperature override must have shape "
                    f"[B, H]={tuple(learned_temperature.shape[:2])}, got "
                    f"{tuple(override.shape)}."
                )
            override = override.unsqueeze(-1).unsqueeze(-1)

        try:
            override = torch.broadcast_to(
                override,
                learned_temperature.shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                "Temperature override is not broadcastable to "
                f"{tuple(learned_temperature.shape)}."
            ) from exc

        if not torch.isfinite(override).all():
            raise ValueError("Temperature override contains non-finite values.")
        if (override <= 0).any():
            raise ValueError("All temperature values must be strictly positive.")

        return override

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        temperature_override=None,
    ) -> torch.Tensor:
        batch_size, query_length, _ = query.shape

        q = self._split_heads(self.q_projection(query))
        k = self._split_heads(self.k_projection(key))
        v = self._split_heads(self.v_projection(value))

        # Query-Key compatibility: [B, heads].
        q_summary = F.normalize(q.mean(dim=2), p=2, dim=-1)
        k_summary = F.normalize(k.mean(dim=2), p=2, dim=-1)
        similarity = (q_summary * k_summary).sum(dim=-1)

        positive_scale = F.softplus(self.raw_temperature_scale) + 1e-6
        temperature_gate = torch.sigmoid(
            positive_scale * similarity + self.temperature_bias.unsqueeze(0)
        )
        temperature_factor = (
            self.min_temp_factor
            + (self.max_temp_factor - self.min_temp_factor) * temperature_gate
        )

        # Learned temperature: [B, heads, 1, 1].
        learned_temperature = (
            math.sqrt(self.head_dim)
            * temperature_factor.unsqueeze(-1).unsqueeze(-1)
        )
        temperature = self._prepare_temperature_override(
            temperature_override,
            learned_temperature,
        )

        raw_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_logits = raw_scores / temperature
        attention_probabilities = F.softmax(attention_logits, dim=-1)

        # Keep the normalized, pre-dropout probabilities for entropy analysis.
        attention_used = self.attention_dropout(attention_probabilities)
        attended = torch.matmul(attention_used, v)
        attended = (
            attended.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, query_length, self.feature_dim)
        )

        self.last_raw_scores = raw_scores.detach()
        self.last_attention_logits = attention_logits.detach()
        self.last_attention = attention_probabilities.detach()
        self.last_similarity = similarity.detach()
        self.last_temperature_factor = temperature_factor.detach()
        self.last_learned_temperature = learned_temperature.detach()
        self.last_temperature = temperature.detach()

        return self.output_projection(attended)
class LiquidDepthUpdate(nn.Module):
    """Liquid memory update across stacked cross-modal interaction layers."""

    def __init__(
        self,
        hidden_dim: int,
        tau: float = 0.15,
        gamma: float = 0.6,
        dropout: float = 0.1,
    ):
        super().__init__()
        tau = min(max(float(tau), 1e-4), 1.0 - 1e-4)
        self.raw_tau = nn.Parameter(torch.tensor(math.log(tau / (1.0 - tau))))
        self.gamma = float(gamma)

        self.attention_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        query_features: torch.Tensor,
        attention_features: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_state is None:
            hidden_state = torch.zeros_like(query_features)

        projected = self.attention_projection(attention_features)
        update_gate = torch.sigmoid(
            self.gate(torch.cat([query_features, attention_features, hidden_state], dim=-1))
        )

        tau = torch.sigmoid(self.raw_tau)
        candidate_state = (1.0 - tau) * hidden_state + tau * torch.tanh(projected)
        new_hidden = (
            update_gate * candidate_state
            + (1.0 - update_gate) * (self.gamma * hidden_state)
        )
        output = self.output_projection(torch.cat([projected, new_hidden], dim=-1))
        return output, new_hidden


class DirectionalCrossModalUpdate(nn.Module):
    """One independent direction of cross-modal feature exchange."""

    def __init__(self, feature_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = AdaptiveTemperatureCrossAttention(
            feature_dim,
            num_heads,
            dropout=dropout,
        )
        self.directional_gate = nn.Linear(feature_dim * 2, feature_dim)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.liquid_update = LiquidDepthUpdate(feature_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.norm3 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        temperature_override=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_output = self.attention(
            query,
            key_value,
            key_value,
            temperature_override=temperature_override,
        )
        gate = torch.sigmoid(
            self.directional_gate(torch.cat([query, attention_output], dim=-1))
        )
        gated = gate * attention_output + (1.0 - gate) * query
        x = self.norm1(gated)

        liquid_output, new_hidden = self.liquid_update(
            x,
            attention_output,
            hidden_state,
        )
        x = self.norm2(x + self.dropout(liquid_output))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x, new_hidden
class BidirectionalCrossModalLayer(nn.Module):
    """Parallel RNA-to-protein and protein-to-RNA interaction layer."""

    def __init__(self, feature_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.rna_to_protein = DirectionalCrossModalUpdate(
            feature_dim,
            num_heads,
            dropout,
        )
        self.protein_to_rna = DirectionalCrossModalUpdate(
            feature_dim,
            num_heads,
            dropout,
        )

    def forward(
        self,
        protein_tokens: torch.Tensor,
        rna_tokens: torch.Tensor,
        protein_hidden: Optional[torch.Tensor] = None,
        rna_hidden: Optional[torch.Tensor] = None,
        temperature_overrides: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        overrides = temperature_overrides or {}

        # Both directions use the same previous-layer representations.
        new_protein, new_protein_hidden = self.rna_to_protein(
            protein_tokens,
            rna_tokens,
            protein_hidden,
            temperature_override=overrides.get("rna_to_protein"),
        )
        new_rna, new_rna_hidden = self.protein_to_rna(
            rna_tokens,
            protein_tokens,
            rna_hidden,
            temperature_override=overrides.get("protein_to_rna"),
        )
        return new_protein, new_rna, new_protein_hidden, new_rna_hidden


# -----------------------------------------------------------------------------
# Multi-view fusion
# -----------------------------------------------------------------------------
class MultiViewFusionTransformer(nn.Module):
    """Fuse five view vectors using a short non-degenerate Transformer."""

    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        num_views: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.view_embedding = nn.Parameter(
            torch.empty(1, num_views, feature_dim)
        )
        nn.init.normal_(self.view_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        x = views + self.view_embedding[:, : views.shape[1], :]
        x = self.encoder(x)
        return self.norm(x.mean(dim=1))


# -----------------------------------------------------------------------------
# Shared modality encoders
# -----------------------------------------------------------------------------
class SharedModalityBackbone(nn.Module):
    """Encode composition features and foundation-model semantics in separate branches.

    Composition features are tokenized and pooled independently. RNA-FM and
    ESM-2 embeddings are encoded by lightweight semantic MLPs, explicitly
    interacted, and injected into latent tokens through token-wise gates.
    """

    def __init__(
        self,
        num_protein_features: int,
        num_rna_features: int,
        rna_fm_dim: int,
        esm2_dim: int,
        token_dim: int,
        num_heads: int,
        num_latents: int,
        dropout: float,
        semantic_hidden_dim: int = 128,
    ):
        super().__init__()
        self.token_dim = int(token_dim)

        # Composition-only branch. Foundation embeddings do not participate in
        # this latent-pooling attention and therefore cannot be diluted by the
        # 256 RNA or 400 protein scalar-feature tokens.
        self.protein_tokenizer = ContinuousFeatureTokenizer(
            num_protein_features,
            token_dim,
            dropout,
        )
        self.rna_tokenizer = ContinuousFeatureTokenizer(
            num_rna_features,
            token_dim,
            dropout,
        )
        self.protein_pooler = LatentTokenPooler(
            token_dim,
            num_heads,
            num_latents,
            dropout,
        )
        self.rna_pooler = LatentTokenPooler(
            token_dim,
            num_heads,
            num_latents,
            dropout,
        )
        self.protein_encoder = DualPathDynamicFeatureProcessor(token_dim, dropout)
        self.rna_encoder = nn.Sequential(
            RNAFeatureEncoder(token_dim, num_heads, dropout),
            DualPathDynamicFeatureProcessor(token_dim, dropout),
        )

        # Independent lightweight semantic encoders requested for E3.
        self.rna_fm_encoder = nn.Sequential(
            nn.Linear(rna_fm_dim, semantic_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(semantic_hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.esm2_encoder = nn.Sequential(
            nn.Linear(esm2_dim, semantic_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(semantic_hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
        )

        # Pair-level semantic interaction. Product and absolute difference give
        # the lightweight branch explicit compatibility and distance signals.
        self.semantic_pair_interaction = nn.Sequential(
            nn.Linear(token_dim * 4, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim * 2),
            nn.LayerNorm(token_dim * 2),
        )

        # Token-wise gated residual injection preserves composition-token
        # diversity while allowing the semantic branch to modulate each token.
        self.protein_semantic_gate = nn.Linear(token_dim * 2, token_dim)
        self.rna_semantic_gate = nn.Linear(token_dim * 2, token_dim)
        self.protein_fusion_norm = nn.LayerNorm(token_dim)
        self.rna_fusion_norm = nn.LayerNorm(token_dim)
        self.semantic_dropout = nn.Dropout(dropout)

        # Non-persistent diagnostics from the most recent forward pass.
        self.last_protein_semantic_gate = None
        self.last_rna_semantic_gate = None

    def _semantic_contexts(
        self,
        rna_fm_features: torch.Tensor,
        esm2_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rna_semantic = self.rna_fm_encoder(rna_fm_features)
        protein_semantic = self.esm2_encoder(esm2_features)

        pair_features = torch.cat(
            [
                protein_semantic,
                rna_semantic,
                protein_semantic * rna_semantic,
                torch.abs(protein_semantic - rna_semantic),
            ],
            dim=-1,
        )
        paired_context = self.semantic_pair_interaction(pair_features)
        protein_context, rna_context = paired_context.chunk(2, dim=-1)
        return protein_context, rna_context

    def _gated_context_injection(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
        gate_layer: nn.Linear,
        norm_layer: nn.LayerNorm,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        expanded_context = context.unsqueeze(1).expand(-1, tokens.shape[1], -1)
        gate = torch.sigmoid(
            gate_layer(torch.cat([tokens, expanded_context], dim=-1))
        )
        fused = norm_layer(
            tokens + self.semantic_dropout(gate * expanded_context)
        )
        return fused, gate

    def encode_modalities(
        self,
        protein_features: torch.Tensor,
        rna_features: torch.Tensor,
        rna_fm_features: torch.Tensor,
        esm2_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        protein_tokens = self.protein_tokenizer(protein_features)
        rna_tokens = self.rna_tokenizer(rna_features)

        protein_latents = self.protein_pooler(protein_tokens)
        rna_latents = self.rna_pooler(rna_tokens)
        protein_encoded = self.protein_encoder(protein_latents)
        rna_encoded = self.rna_encoder(rna_latents)

        protein_context, rna_context = self._semantic_contexts(
            rna_fm_features,
            esm2_features,
        )
        protein_fused, protein_gate = self._gated_context_injection(
            protein_encoded,
            protein_context,
            self.protein_semantic_gate,
            self.protein_fusion_norm,
        )
        rna_fused, rna_gate = self._gated_context_injection(
            rna_encoded,
            rna_context,
            self.rna_semantic_gate,
            self.rna_fusion_norm,
        )

        self.last_protein_semantic_gate = protein_gate.detach()
        self.last_rna_semantic_gate = rna_gate.detach()
        return protein_fused, rna_fused


class _InputFeatureMixin:
    """Resolve composition, RNA-FM, and ESM-2 indices from column names."""

    def _setup_feature_indices(
        self,
        feat_size: Mapping[str, int],
        dnn_feature_columns: Sequence[Tuple[str, str]],
    ) -> None:
        names = _ordered_feature_names(feat_size, dnn_feature_columns)
        name_to_index = {name: index for index, name in enumerate(names)}

        def matches(name: str, prefix: str) -> bool:
            upper = name.upper()
            return upper.startswith(prefix) and upper[len(prefix):].isdigit()

        # Exact patterns avoid treating RF1 as an ordinary R* composition column.
        protein_names = sorted(
            [name for name in names if matches(name, "P")],
            key=_numeric_suffix,
        )
        rna_names = sorted(
            [name for name in names if matches(name, "R")],
            key=_numeric_suffix,
        )
        rna_fm_names = sorted(
            [name for name in names if matches(name, "RF")],
            key=_numeric_suffix,
        )
        esm2_names = sorted(
            [name for name in names if matches(name, "E")],
            key=_numeric_suffix,
        )

        if not protein_names or not rna_names:
            raise ValueError(
                "Could not identify composition columns P1... and R1...."
            )
        if not rna_fm_names or not esm2_names:
            raise ValueError(
                "Could not identify RNA-FM columns RF1... or ESM-2 columns E1...."
            )

        def index_tensor(selected_names: Sequence[str]) -> torch.Tensor:
            return torch.tensor(
                [name_to_index[name] for name in selected_names],
                dtype=torch.long,
            )

        self.register_buffer(
            "protein_indices",
            index_tensor(protein_names),
            persistent=False,
        )
        self.register_buffer(
            "rna_indices",
            index_tensor(rna_names),
            persistent=False,
        )
        self.register_buffer(
            "rna_fm_indices",
            index_tensor(rna_fm_names),
            persistent=False,
        )
        self.register_buffer(
            "esm2_indices",
            index_tensor(esm2_names),
            persistent=False,
        )

        self.num_protein_features = len(protein_names)
        self.num_rna_features = len(rna_names)
        self.rna_fm_dim = len(rna_fm_names)
        self.esm2_dim = len(esm2_names)

    def _split_inputs(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(
                f"Expected two-dimensional input [B, F], got {tuple(x.shape)}"
            )
        protein = x.index_select(1, self.protein_indices)
        rna = x.index_select(1, self.rna_indices)
        rna_fm = x.index_select(1, self.rna_fm_indices)
        esm2 = x.index_select(1, self.esm2_indices)
        return protein, rna, rna_fm, esm2


# -----------------------------------------------------------------------------
# Warmup and full models
# -----------------------------------------------------------------------------
class DaLNPI_warmup(nn.Module, _InputFeatureMixin):
    """Fold-specific warmup model for the shared modality encoders."""

    def __init__(
        self,
        feat_size: Mapping[str, int],
        embedding_size: int,
        dnn_feature_columns: Sequence[Tuple[str, str]],
        att_layer_num: int = 2,
        num_heads: int = 4,
        num_latents: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        del att_layer_num  # Kept only for constructor compatibility.

        token_dim = int(embedding_size)
        if token_dim < num_heads or token_dim % num_heads != 0:
            raise ValueError(
                "For the revised model, embedding_size is the token dimension and "
                "must be divisible by num_heads. Recommended: embedding_size=64."
            )

        self._setup_feature_indices(feat_size, dnn_feature_columns)
        self.backbone = SharedModalityBackbone(
            self.num_protein_features,
            self.num_rna_features,
            self.rna_fm_dim,
            self.esm2_dim,
            token_dim,
            num_heads,
            num_latents,
            dropout,
        )
        self.warmup_predictor = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(token_dim * 2, 1),
        )

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return pre-sigmoid logits without changing the training architecture."""
        x = x.to(next(self.parameters()).device, dtype=torch.float32)
        (
            protein_features,
            rna_features,
            rna_fm_features,
            esm2_features,
        ) = self._split_inputs(x)
        protein_tokens, rna_tokens = self.backbone.encode_modalities(
            protein_features,
            rna_features,
            rna_fm_features,
            esm2_features,
        )
        protein_vector = protein_tokens.mean(dim=1)
        rna_vector = rna_tokens.mean(dim=1)
        return self.warmup_predictor(
            torch.cat([protein_vector, rna_vector], dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))

class DaLNPI(nn.Module, _InputFeatureMixin):
    """Complete LNN-DACA model enhanced with RNA-FM and ESM-2 sequence embeddings.

    The public ``forward`` method retains the original probability output used by
    the training scripts. Additional methods expose logits, the six view vectors,
    and attention state for post-hoc interpretability analyses without introducing
    new trainable parameters.
    """

    VIEW_NAMES = (
        "post_protein",
        "post_rna",
        "initial_protein",
        "initial_rna",
        "raw_composition",
        "topology",
    )

    ARCHITECTURE_COMPONENTS = (
        "continuous_feature_tokenization",
        "latent_token_pooling",
        "liquid_dual_path_feature_processing",
        "independent_rnafm_esm2_semantic_encoding",
        "pair_semantic_interaction",
        "tokenwise_semantic_gating",
        "bidirectional_adaptive_temperature_cross_attention",
        "directional_cross_modal_gating",
        "liquid_depth_update",
        "fold_safe_lightweight_topology_view",
        "six_view_transformer_fusion",
    )

    def __init__(
        self,
        feat_size: Mapping[str, int],
        embedding_size: int,
        dnn_feature_columns: Sequence[Tuple[str, str]],
        att_layer_num: int = 2,
        num_heads: int = 4,
        num_latents: int = 16,
        dropout: float = 0.1,
        topology_feature_dim: int = 10,
    ):
        super().__init__()
        token_dim = int(embedding_size)
        if token_dim < num_heads or token_dim % num_heads != 0:
            raise ValueError(
                "For the revised model, embedding_size is the token dimension and "
                "must be divisible by num_heads. Recommended: embedding_size=64."
            )

        self._setup_feature_indices(feat_size, dnn_feature_columns)
        self.backbone = SharedModalityBackbone(
            self.num_protein_features,
            self.num_rna_features,
            self.rna_fm_dim,
            self.esm2_dim,
            token_dim,
            num_heads,
            num_latents,
            dropout,
        )

        self.cross_modal_layers = nn.ModuleList(
            [
                BidirectionalCrossModalLayer(
                    token_dim,
                    num_heads,
                    dropout,
                )
                for _ in range(att_layer_num)
            ]
        )

        self.raw_feature_projection = nn.Sequential(
            nn.Linear(self.num_protein_features + self.num_rna_features, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim),
        )
        self.topology_feature_dim = int(topology_feature_dim)
        if self.topology_feature_dim <= 0:
            raise ValueError("topology_feature_dim must be positive.")
        self.topology_projection = nn.Sequential(
            nn.LayerNorm(self.topology_feature_dim),
            nn.Linear(self.topology_feature_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.multi_view_fusion = MultiViewFusionTransformer(
            token_dim,
            num_heads,
            num_views=6,
            dropout=dropout,
        )
        self.predictor = nn.Sequential(
            nn.Linear(token_dim * 5, token_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(token_dim * 2, 1),
        )

    def extract_view_vectors(
        self,
        x: torch.Tensor,
        temperature_overrides: Optional[
            Sequence[Mapping[str, torch.Tensor]]
        ] = None,
        return_tokens: bool = False,
        topology_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode an input batch and return the six vectors used for fusion.

        ``temperature_overrides`` may be a sequence with one mapping per
        cross-modal layer. Each mapping can contain ``rna_to_protein`` and/or
        ``protein_to_rna``. Values denote actual positive attention temperatures.
        Omitting this argument reproduces the original model exactly.
        """
        x = x.to(next(self.parameters()).device, dtype=torch.float32)
        (
            protein_features,
            rna_features,
            rna_fm_features,
            esm2_features,
        ) = self._split_inputs(x)

        initial_protein_tokens, initial_rna_tokens = self.backbone.encode_modalities(
            protein_features,
            rna_features,
            rna_fm_features,
            esm2_features,
        )
        protein_tokens = initial_protein_tokens
        rna_tokens = initial_rna_tokens
        protein_hidden: Optional[torch.Tensor] = None
        rna_hidden: Optional[torch.Tensor] = None

        if (
            temperature_overrides is not None
            and len(temperature_overrides) != len(self.cross_modal_layers)
        ):
            raise ValueError(
                "temperature_overrides must contain one mapping per "
                f"cross-modal layer ({len(self.cross_modal_layers)} required)."
            )

        for layer_index, layer in enumerate(self.cross_modal_layers):
            layer_overrides = (
                None
                if temperature_overrides is None
                else temperature_overrides[layer_index]
            )
            protein_tokens, rna_tokens, protein_hidden, rna_hidden = layer(
                protein_tokens,
                rna_tokens,
                protein_hidden,
                rna_hidden,
                temperature_overrides=layer_overrides,
            )

        if topology_features is None:
            topology_features = x.new_zeros(
                x.shape[0],
                self.topology_feature_dim,
            )
        else:
            topology_features = topology_features.to(
                device=x.device,
                dtype=torch.float32,
            )
            if (
                topology_features.ndim != 2
                or topology_features.shape[0] != x.shape[0]
                or topology_features.shape[1] != self.topology_feature_dim
            ):
                raise ValueError(
                    "Expected topology features with shape "
                    f"[B, {self.topology_feature_dim}], got "
                    f"{tuple(topology_features.shape)}."
                )

        view_dict: Dict[str, torch.Tensor] = {
            "post_protein": protein_tokens.mean(dim=1),
            "post_rna": rna_tokens.mean(dim=1),
            "initial_protein": initial_protein_tokens.mean(dim=1),
            "initial_rna": initial_rna_tokens.mean(dim=1),
            "raw_composition": self.raw_feature_projection(
                torch.cat([protein_features, rna_features], dim=-1)
            ),
            "topology": self.topology_projection(topology_features),
        }

        if return_tokens:
            view_dict.update(
                {
                    "initial_protein_tokens": initial_protein_tokens,
                    "initial_rna_tokens": initial_rna_tokens,
                    "post_protein_tokens": protein_tokens,
                    "post_rna_tokens": rna_tokens,
                }
            )

        return view_dict

    def logits_from_views(
        self,
        view_dict: Mapping[str, torch.Tensor],
        return_intermediates: bool = False,
    ):
        """Compute prediction logits from externally supplied view vectors.

        This interface supports inference-time view permutation or replacement.
        All six required views must have shape [B, token_dim]. Extra dictionary
        entries are ignored.
        """
        missing = [name for name in self.VIEW_NAMES if name not in view_dict]
        if missing:
            raise KeyError(f"Missing required view vectors: {missing}")

        device = next(self.parameters()).device
        prepared = {
            name: view_dict[name].to(device=device, dtype=torch.float32)
            for name in self.VIEW_NAMES
        }

        batch_sizes = {tensor.shape[0] for tensor in prepared.values()}
        if len(batch_sizes) != 1:
            raise ValueError("All view vectors must have the same batch size.")
        if any(tensor.ndim != 2 for tensor in prepared.values()):
            raise ValueError("Every view vector must have shape [B, D].")

        views = torch.stack(
            [prepared[name] for name in self.VIEW_NAMES],
            dim=1,
        )
        fused_vector = self.multi_view_fusion(views)

        final_features = torch.cat(
            [
                prepared["post_protein"],
                prepared["post_rna"],
                prepared["initial_protein"],
                prepared["initial_rna"],
                fused_vector,
            ],
            dim=-1,
        )
        logits = self.predictor(final_features)

        if return_intermediates:
            return {
                "logits": logits,
                "fused_vector": fused_vector,
                "stacked_views": views,
                **prepared,
            }
        return logits

    def forward_logits(
        self,
        x: torch.Tensor,
        temperature_overrides: Optional[
            Sequence[Mapping[str, torch.Tensor]]
        ] = None,
        return_intermediates: bool = False,
        topology_features: Optional[torch.Tensor] = None,
    ):
        """Return pre-sigmoid logits, optionally with all view representations."""
        view_dict = self.extract_view_vectors(
            x,
            temperature_overrides=temperature_overrides,
            return_tokens=False,
            topology_features=topology_features,
        )
        return self.logits_from_views(
            view_dict,
            return_intermediates=return_intermediates,
        )

    def forward_with_intermediates(
        self,
        x: torch.Tensor,
        temperature_overrides: Optional[
            Sequence[Mapping[str, torch.Tensor]]
        ] = None,
        topology_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return probabilities together with logits and the six view vectors."""
        outputs = self.forward_logits(
            x,
            temperature_overrides=temperature_overrides,
            return_intermediates=True,
            topology_features=topology_features,
        )
        outputs["probabilities"] = torch.sigmoid(outputs["logits"])
        return outputs

    def forward(
        self,
        x: torch.Tensor,
        temperature_overrides: Optional[
            Sequence[Mapping[str, torch.Tensor]]
        ] = None,
        topology_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.forward_logits(
            x,
            temperature_overrides=temperature_overrides,
            return_intermediates=False,
            topology_features=topology_features,
        )
        return torch.sigmoid(logits)

    def get_interpretability_state(self) -> Dict[str, object]:
        """Return tensors generated during the most recent forward pass.

        Cross-attention maps describe associations among learned latent tokens
        pooled from composition features and one foundation-model semantic token
        per modality; they are not residue- or nucleotide-level contact maps.
        """
        cross_attention = []
        attention_logits = []
        raw_scores = []
        temperatures = []
        learned_temperatures = []
        temperature_factors = []
        similarities = []

        for layer in self.cross_modal_layers:
            direction_modules = {
                "rna_to_protein": layer.rna_to_protein.attention,
                "protein_to_rna": layer.protein_to_rna.attention,
            }

            cross_attention.append(
                {
                    name: module.last_attention
                    for name, module in direction_modules.items()
                }
            )
            attention_logits.append(
                {
                    name: module.last_attention_logits
                    for name, module in direction_modules.items()
                }
            )
            raw_scores.append(
                {
                    name: module.last_raw_scores
                    for name, module in direction_modules.items()
                }
            )
            temperatures.append(
                {
                    name: module.last_temperature
                    for name, module in direction_modules.items()
                }
            )
            learned_temperatures.append(
                {
                    name: module.last_learned_temperature
                    for name, module in direction_modules.items()
                }
            )
            temperature_factors.append(
                {
                    name: module.last_temperature_factor
                    for name, module in direction_modules.items()
                }
            )
            similarities.append(
                {
                    name: module.last_similarity
                    for name, module in direction_modules.items()
                }
            )

        return {
            "protein_pooling_attention": self.backbone.protein_pooler.last_attention,
            "rna_pooling_attention": self.backbone.rna_pooler.last_attention,
            "cross_attention": cross_attention,
            "attention_logits": attention_logits,
            "raw_scores": raw_scores,
            "temperatures": temperatures,
            "learned_temperatures": learned_temperatures,
            "temperature_factors": temperature_factors,
            "similarities": similarities,
            "protein_semantic_gate": self.backbone.last_protein_semantic_gate,
            "rna_semantic_gate": self.backbone.last_rna_semantic_gate,
            "topology_feature_dim": self.topology_feature_dim,
        }

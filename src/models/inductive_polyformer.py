"""
Inductive 3D-Aware PolyFormer:
Higher-Order Polypharmacy Risk Prediction without Categorical Drug Lookups.

Key Architectural Components:
1. Pure Inductive Input: Encodes 3D conformer geometry and 2D chemical fingerprints.
2. Pairwise Synergy Layer: Explicitly models all pairwise combinations within the cocktail
   to ensure dangerous pairs (e.g., Warfarin + Aspirin) are never washed out.
3. Set Self-Attention: TransformerEncoder layers model higher-order (3+ drug) collective effects.
4. Phenotype Cross-Attention: Side effect queries the contextualized drug set.
5. Multi-Signal Fusion Head: Combines higher-order attention, pairwise synergy, and side-effect query.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.conformer_3d import TOTAL_DRUG_FEAT_DIM


class PairwiseSynergyLayer(nn.Module):
    """
    Computes pairwise interaction signals between every pair of drugs in a prescription.
    For drugs i and j:
        x_ij = Linear(concat([d_i, d_j, |d_i - d_j|, d_i * d_j]))
    Masked pooling yields a dedicated pairwise risk representation.
    """
    def __init__(self, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, drug_feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        drug_feats: [B, K, d_model]
        mask:       [B, K] (1 for real drug, 0 for pad)
        returns:    [B, d_model]
        """
        B, K, D = drug_feats.shape
        if K < 2:
            return torch.zeros((B, D), device=drug_feats.device, dtype=drug_feats.dtype)

        # Broadcast into pairs [B, K, K, D]
        d_i = drug_feats.unsqueeze(2).expand(B, K, K, D)
        d_j = drug_feats.unsqueeze(1).expand(B, K, K, D)

        diff = torch.abs(d_i - d_j)
        prod = d_i * d_j
        pair_input = torch.cat([d_i, d_j, diff, prod], dim=-1)  # [B, K, K, 4*D]
        pair_repr = self.proj(pair_input)  # [B, K, K, D]

        # Valid pair mask: both drugs must be real and i != j
        m_i = mask.unsqueeze(2).expand(B, K, K)  # [B, K, K]
        m_j = mask.unsqueeze(1).expand(B, K, K)
        pair_mask = (m_i * m_j).bool()

        # Zero out diagonal (self-pairs)
        diag_mask = torch.eye(K, device=drug_feats.device, dtype=torch.bool).unsqueeze(0).expand(B, K, K)
        pair_mask = pair_mask & (~diag_mask)

        # Masked average pooling
        mask_expanded = pair_mask.unsqueeze(-1).float()
        sum_pooled = (pair_repr * mask_expanded).sum(dim=(1, 2))  # [B, D]
        count = mask_expanded.sum(dim=(1, 2)).clamp(min=1.0)
        return sum_pooled / count


class InductivePolyFormer(nn.Module):
    """
    Purely Inductive PolyFormer for polypharmacy risk prediction.
    Zero transductive Drug ID embeddings: fully evaluated on molecular descriptors.
    """
    def __init__(
        self,
        num_ses: int,
        in_drug_dim: int = TOTAL_DRUG_FEAT_DIM,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Molecular Feature Projection
        self.drug_proj = nn.Sequential(
            nn.Linear(in_drug_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # 2. Side Effect Representation (Categorical lookup + semantic projection hook)
        self.se_emb = nn.Embedding(num_ses, d_model)

        # 3. Explicit Pairwise Synergy Sub-Layer
        self.pairwise_layer = PairwiseSynergyLayer(d_model=d_model, dropout=dropout)

        # 4. Set Transformer Encoder (High-Order Self-Attention)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=False,
        )
        self.set_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 5. Phenotype Cross-Attention (SE queries the drug set)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        # 6. Multi-Signal Prediction Classifier
        # Concatenates: [Higher-order Combo (d_model), Pairwise Synergy (d_model), SE Token (d_model)]
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.LayerNorm(2 * d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        drug_features: torch.Tensor,
        mask: torch.Tensor,
        se_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        drug_features: [B, K, in_drug_dim] (3D conformer + 2D fingerprint features)
        mask:          [B, K] (1 for real drug, 0 for padding)
        se_idx:        [B] (Side effect integer label)
        
        returns:
            logits: [B] Interaction risk logit
        """
        B, K, _ = drug_features.shape

        # 1. Project molecular features to model dimension
        drug_embed = self.drug_proj(drug_features)  # [B, K, d_model]

        # 2. Pairwise Synergy Representation
        pair_synergy = self.pairwise_layer(drug_embed, mask)  # [B, d_model]

        # 3. Set Transformer Self-Attention
        key_padding = (mask == 0)
        drug_ctx = self.set_encoder(drug_embed, src_key_padding_mask=key_padding)  # [B, K, d_model]

        # 4. Phenotype Cross-Attention
        q = self.se_emb(se_idx).unsqueeze(1)  # [B, 1, d_model]
        attn_out, _ = self.cross_attn(
            query=q, key=drug_ctx, value=drug_ctx, key_padding_mask=key_padding
        )

        combo = attn_out.squeeze(1)  # [B, d_model]
        se_token = q.squeeze(1)      # [B, d_model]

        # 5. Multi-Signal Concatenation & Risk Prediction
        fusion_vec = torch.cat([combo, pair_synergy, se_token], dim=-1)  # [B, 3 * d_model]
        logits = self.mlp(fusion_vec).squeeze(-1)
        return logits

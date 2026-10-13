"""
PolyFormer: Set Transformer architecture for higher-order polypharmacy risk prediction.
"""

import torch
import torch.nn as nn
from src.models.encoders import SmilesEncoder


class PolyFormerLearnableSE(nn.Module):
    """
    PolyFormer with learnable side-effect embeddings.
    
    1. Drug Feature Construction:
       drug_feat = Linear(concat([ID_emb, SmilesEncoder(smiles)]))
    2. Self-Attention:
       TransformerEncoder allows drugs within each prescription set to attend to one another.
    3. Cross-Attention:
       Side effect query attends over the contextualized drug set representations.
    4. Prediction Head:
       MLP(concat([combo_vector, se_token])) -> Risk logit.
    """
    def __init__(
        self,
        num_drugs: int,
        smiles_vocab_size: int,
        num_ses: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.drug_id_emb = nn.Embedding(num_drugs, 64)
        self.smiles_enc = SmilesEncoder(smiles_vocab_size, embed_dim=64, out_dim=64)
        self.drug_proj = nn.Linear(128, d_model)

        self.se_emb = nn.Embedding(num_ses, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True,
            activation="relu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=0.1, batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        drugs_pad: torch.Tensor,
        mask: torch.Tensor,
        se_idx: torch.Tensor,
        smiles_tok_all: torch.Tensor,
    ) -> torch.Tensor:
        """
        drugs_pad:      [B, K] drug indices, -1 padded
        mask:           [B, K] 1 real drug, 0 pad
        se_idx:         [B] side-effect index
        smiles_tok_all: [num_drugs, max_smiles_len]
        
        returns:
            logits: [B] interaction risk logits
        """
        B, K = drugs_pad.shape
        drugs_safe = drugs_pad.clamp(min=0)

        # 1. Drug ID embedding
        did = self.drug_id_emb(drugs_safe)  # [B, K, 64]

        # 2. Vectorized SMILES encoding for unique drugs in batch
        flat = drugs_safe.reshape(-1)
        uniq, inv = torch.unique(flat, return_inverse=True)
        smiles_tokens = smiles_tok_all[uniq]
        smiles_uniq = self.smiles_enc(smiles_tokens)
        smiles_feat = smiles_uniq[inv].reshape(B, K, -1)  # [B, K, 64]

        # Concat ID + chemical representation
        drug_feat = torch.cat([did, smiles_feat], dim=-1)  # [B, K, 128]
        drug_feat = self.drug_proj(drug_feat)              # [B, K, d_model]

        key_padding = (mask == 0)  # True where padded

        # 3. Drug-drug set self-attention
        drug_ctx = self.encoder(drug_feat, src_key_padding_mask=key_padding)  # [B, K, d_model]

        # 4. Side effect cross-attention querying the drug set
        q = self.se_emb(se_idx).unsqueeze(1)  # [B, 1, d_model]
        attn_out, _ = self.cross_attn(
            query=q, key=drug_ctx, value=drug_ctx, key_padding_mask=key_padding
        )

        combo = attn_out.squeeze(1)  # [B, d_model]
        se_token = q.squeeze(1)      # [B, d_model]

        # 5. Risk logit output
        logits = self.mlp(torch.cat([combo, se_token], dim=-1)).squeeze(-1)
        return logits

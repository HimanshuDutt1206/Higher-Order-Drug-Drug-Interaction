"""
HGNN-SA: Hypergraph Neural Network with SMILES features and attention pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders import SmilesEncoder

try:
    from torch_geometric.nn import HypergraphConv
    from torch_geometric.utils import scatter, softmax
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False


class HGNN_SA(nn.Module):
    """
    Hypergraph Neural Network baseline with SMILES CNN node features
    and hyperedge attention pooling.
    """
    def __init__(self, num_drugs: int, num_ses: int, smiles_vocab_size: int):
        super().__init__()
        if not _PYG_AVAILABLE:
            raise ImportError(
                "torch_geometric is required to instantiate HGNN_SA. "
                "Please install torch-geometric and pyg-lib."
            )

        self.drug_id_emb = nn.Embedding(num_drugs, 64)
        self.smiles_enc = SmilesEncoder(smiles_vocab_size, embed_dim=64, out_dim=64)
        self.node_proj = nn.Linear(128, 128)

        self.se_emb = nn.Embedding(num_ses, 128)

        self.conv1 = HypergraphConv(128, 128)
        self.conv2 = HypergraphConv(128, 128)

        self.attn_layer = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        self.mlp = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )

    def forward(
        self,
        hyperedge_index: torch.Tensor,
        se_indices: torch.Tensor,
        all_smiles_tokens: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Chemical + ID Features
        smiles_feats = self.smiles_enc(all_smiles_tokens)
        node_feats = F.relu(
            self.node_proj(torch.cat([self.drug_id_emb.weight, smiles_feats], dim=1))
        )

        # 2. Hypergraph Message Passing
        x = F.relu(self.conv1(node_feats, hyperedge_index))
        x = F.relu(self.conv2(x, hyperedge_index))

        # 3. Hyperedge Attention Pooling
        drugs = x[hyperedge_index[0]]
        attn_probs = softmax(self.attn_layer(drugs), hyperedge_index[1], dim=0)
        combo_vector = scatter(
            drugs * attn_probs, hyperedge_index[1], dim=0, reduce="sum"
        )

        # 4. Predict Risk Logit
        out = torch.cat([combo_vector, self.se_emb(se_indices)], dim=1)
        return self.mlp(out).squeeze(-1)

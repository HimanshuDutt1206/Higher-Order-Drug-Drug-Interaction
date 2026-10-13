"""
Molecular and SMILES feature encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmilesEncoder(nn.Module):
    """
    1D-CNN Encoder for character-tokenized SMILES representations.
    
    Architecture:
        Embedding(vocab_size, embed_dim, padding_idx=0)
        Conv1d(embed_dim -> 128, kernel=3, pad=1) -> ReLU
        Conv1d(128 -> out_dim, kernel=3, pad=1) -> ReLU
        Global Max Pooling over sequence length -> [B, out_dim]
    """
    def __init__(self, vocab_size: int, embed_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, out_dim, kernel_size=3, padding=1)

    def forward(self, smiles_tokens: torch.Tensor) -> torch.Tensor:
        """
        smiles_tokens: [B, L]
        returns: [B, out_dim]
        """
        x = self.embed(smiles_tokens).transpose(1, 2)  # [B, embed_dim, L]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(-1)  # [B, out_dim]
        return x

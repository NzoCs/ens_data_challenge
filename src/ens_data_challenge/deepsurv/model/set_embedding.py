import torch
import torch.nn as nn
import torch.nn.functional as F

# === Encodeur de sets avec une seule tête d'attention ===
class SetAttention(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        # Projection des éléments
        self.embed = nn.LazyLinear(hidden_dim)
        # Query, Key, Value pour une seule tête
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        # Projection finale
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        """
        x: [batch_size, seq_len, input_dim]  (set d'éléments)
        """
        h = self.embed(x)  # [B, L, H]

        Q = self.q(h)      # [B, L, H]
        K = self.k(h)      # [B, L, H]
        V = self.v(h)      # [B, L, H]

        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (h.size(-1) ** 0.5)  # [B, L, L]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, L, L]

        # Weighted sum
        attended = torch.matmul(attn_weights, V)  # [B, L, H]

        # Pooling sur la dimension des éléments (set)
        pooled = attended.mean(dim=1)  # [B, H]

        # Projection finale
        return self.out(pooled)  # [B, output_dim]

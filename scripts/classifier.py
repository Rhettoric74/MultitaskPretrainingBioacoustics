import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        # x: (B, N, D)
        weights = self.attn(x)                 # (B, N, 1)
        weights = torch.softmax(weights, dim=1)
        pooled = (x * weights).sum(dim=1)      # (B, D)
        return pooled


class BirdClassifierHead(nn.Module):
    def __init__(self, embed_dim, num_classes, dropout=0.2):
        super().__init__()

        self.pool = AttentionPooling(embed_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, N, D)
        pooled = self.pool(x)
        #print("before pooling:", x.shape)
        logits = self.classifier(pooled)
        return logits
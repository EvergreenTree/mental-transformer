from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PairAttentionBlock(nn.Module):
    """Small optional cross-modal block over the image/fMRI pair in a batch row."""

    def __init__(self, latent_dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.norm2 = nn.LayerNorm(latent_dim)

    def forward(self, z_img: torch.Tensor, z_fmri: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.stack([z_img, z_fmri], dim=1)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + attended)
        tokens = self.norm2(tokens + self.ff(tokens))
        return tokens[:, 0], tokens[:, 1]


class MRGSModel(nn.Module):
    def __init__(
        self,
        image_feature_dim: int,
        fmri_dim: int,
        latent_dim: int = 512,
        hidden_dim: int = 1024,
        projector_layers: int = 2,
        encoder_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = False,
        attention_heads: int = 8,
    ) -> None:
        super().__init__()
        self.image_projector = MLP(
            input_dim=image_feature_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=projector_layers,
            dropout=dropout,
        )
        self.fmri_encoder = MLP(
            input_dim=fmri_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            dropout=dropout,
        )
        self.attention = (
            PairAttentionBlock(latent_dim=latent_dim, num_heads=attention_heads, dropout=dropout)
            if use_attention
            else None
        )

    def forward(self, image_features: torch.Tensor, fmri: torch.Tensor) -> dict[str, torch.Tensor]:
        z_img = self.image_projector(image_features)
        z_fmri = self.fmri_encoder(fmri)
        if self.attention is not None:
            z_img, z_fmri = self.attention(z_img, z_fmri)
        return {
            "z_img": F.normalize(z_img, dim=-1),
            "z_fmri": F.normalize(z_fmri, dim=-1),
        }

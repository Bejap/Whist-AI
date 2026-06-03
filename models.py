"""Custom model components for Whist training/inference."""

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from whist_env import (
    NUM_CARDS,
    OBS_CURRENT_TRICK,
    OBS_HAND,
    OBS_PLAYED_BY_PLAYER,
    OBS_SIZE,
)


class TransformerCardExtractor(BaseFeaturesExtractor):
    """Transformer feature extractor over hand + trick card tokens."""

    def __init__(
        self,
        observation_space,
        card_embed_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        features_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)
        self.card_embed = nn.Embedding(NUM_CARDS, card_embed_dim)
        self.type_embed = nn.Embedding(2, card_embed_dim)  # hand / trick
        self.register_buffer("card_ids", torch.arange(NUM_CARDS), persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=card_embed_dim,
            nhead=nhead,
            dim_feedforward=card_embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        context_dim = OBS_SIZE - OBS_HAND - OBS_CURRENT_TRICK
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(card_embed_dim + 128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hand = observations[:, :OBS_HAND]
        trick_start = OBS_HAND + OBS_PLAYED_BY_PLAYER
        trick_end = trick_start + OBS_CURRENT_TRICK
        trick = observations[:, trick_start:trick_end]

        context = torch.cat(
            [observations[:, OBS_HAND:trick_start], observations[:, trick_end:]],
            dim=1,
        )

        batch_size = observations.shape[0]
        card_emb = self.card_embed(self.card_ids).unsqueeze(0).expand(batch_size, -1, -1)
        hand_type = self.type_embed.weight[0].view(1, 1, -1)
        trick_type = self.type_embed.weight[1].view(1, 1, -1)

        hand_tokens = (card_emb + hand_type) * hand.unsqueeze(-1)
        trick_tokens = (card_emb + trick_type) * trick.unsqueeze(-1)
        tokens = torch.cat([hand_tokens, trick_tokens], dim=1)

        padding_mask = torch.cat([hand <= 0, trick <= 0], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        active = (~padding_mask).unsqueeze(-1).float()
        pooled = (encoded * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)

        context_features = self.context_mlp(context)
        return self.proj(torch.cat([pooled, context_features], dim=1))

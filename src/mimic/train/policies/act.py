from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimic.train.policies.base import MimicPolicy


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device).float() * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ACTPolicy(MimicPolicy):
    """Action Chunking Transformer for imitation learning.

    Predicts a chunk of future actions given current state observation.
    Uses a conditional VAE during training for multi-modal action prediction.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_chunk_size: int = 10,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        latent_dim: int = 32,
        dropout: float = 0.1,
        kl_weight: float = 10.0,
    ):
        super().__init__(obs_dim, action_dim, action_chunk_size)
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.kl_weight = kl_weight

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # CVAE encoder (training only) - encodes action sequence to latent
        self.cvae_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.latent_mean = nn.Linear(hidden_dim, latent_dim)
        self.latent_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder - generates action chunk from latent + state
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_emb = SinusoidalPosEmb(hidden_dim)
        self.pos_proj = nn.Linear(hidden_dim, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def encode_latent(self, state: torch.Tensor, actions: torch.Tensor):
        """CVAE encoder: encode state + action sequence to latent distribution."""
        B, T, _ = actions.shape
        state_emb = self.state_encoder(state[:, 0])  # [B, hidden]

        action_emb = self.action_proj(actions)  # [B, T, hidden]
        # Prepend state embedding as CLS token
        cls_token = state_emb.unsqueeze(1)  # [B, 1, hidden]
        encoder_input = torch.cat([cls_token, action_emb], dim=1)  # [B, T+1, hidden]

        encoded = self.cvae_encoder(encoder_input)
        cls_output = encoded[:, 0]  # [B, hidden]

        mean = self.latent_mean(cls_output)
        logvar = self.latent_logvar(cls_output)
        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent + state to action chunk."""
        B = state.shape[0]
        state_emb = self.state_encoder(state[:, 0])  # [B, hidden]
        latent_emb = self.latent_proj(latent)  # [B, hidden]

        # Memory for cross-attention: state + latent
        memory = torch.stack([state_emb, latent_emb], dim=1)  # [B, 2, hidden]

        # Query: positional embeddings for each timestep
        timesteps = torch.arange(self.action_chunk_size, device=state.device).float()
        pos = self.pos_proj(self.pos_emb(timesteps))  # [T, hidden]
        queries = pos.unsqueeze(0).expand(B, -1, -1)  # [B, T, hidden]

        decoded = self.decoder(queries, memory)  # [B, T, hidden]
        actions = self.action_head(decoded)  # [B, T, action_dim]
        return actions

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Training forward pass with CVAE."""
        state = batch["state"]  # [B, T, obs_dim]
        actions = batch["action"]  # [B, T, action_dim]

        # CVAE encode
        mean, logvar = self.encode_latent(state, actions)
        latent = self.reparameterize(mean, logvar)

        # Decode
        pred_actions = self.decode(state, latent)  # [B, T, action_dim]

        # Losses
        T = min(pred_actions.shape[1], actions.shape[1])
        recon_loss = F.mse_loss(pred_actions[:, :T], actions[:, :T])
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        total_loss = recon_loss + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
        }

    def predict(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Inference: predict action chunk from zero latent."""
        state = obs["state"]
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim == 2:
            state = state.unsqueeze(1)  # [B, 1, obs_dim]

        B = state.shape[0]
        latent = torch.zeros(B, self.latent_dim, device=state.device)
        actions = self.decode(state, latent)
        return actions  # [B, T, action_dim]

    def _get_config(self) -> dict:
        return {
            **super()._get_config(),
            "hidden_dim": self.hidden_dim,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "latent_dim": self.latent_dim,
            "dropout": self.dropout,
            "kl_weight": self.kl_weight,
        }

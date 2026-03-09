from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimic.train.policies.base import MimicPolicy


class DiffusionPolicy(MimicPolicy):
    """Diffusion Policy for imitation learning.

    Uses DDPM (Denoising Diffusion Probabilistic Models) to generate
    action chunks by iteratively denoising from Gaussian noise.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_chunk_size: int = 10,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_diffusion_steps: int = 100,
        noise_schedule: str = "cosine",  # "linear" or "cosine"
    ):
        super().__init__(obs_dim, action_dim, action_chunk_size)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_diffusion_steps = n_diffusion_steps
        self.noise_schedule = noise_schedule

        # Noise schedule
        if noise_schedule == "cosine":
            betas = self._cosine_beta_schedule(n_diffusion_steps)
        else:
            betas = torch.linspace(1e-4, 0.02, n_diffusion_steps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Denoiser MLP (predicts noise)
        # Input: noisy_action + state_embedding + time_embedding
        input_dim = action_dim * action_chunk_size + hidden_dim + hidden_dim
        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, action_dim * action_chunk_size))
        self.denoiser = nn.Sequential(*layers)

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = (
            torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def _timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device).float() * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.time_emb(emb)

    def _add_noise(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        noisy = sqrt_alpha * x + sqrt_one_minus_alpha * noise
        return noisy, noise

    def _predict_noise(
        self,
        noisy_action_flat: torch.Tensor,
        state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        state_emb = self.state_encoder(state)
        time_emb = self._timestep_embedding(t)
        x = torch.cat([noisy_action_flat, state_emb, time_emb], dim=-1)
        return self.denoiser(x)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = batch["state"]  # [B, T, obs_dim]
        actions = batch["action"]  # [B, T, action_dim]
        B = state.shape[0]

        # Use first state as condition
        state_cond = state[:, 0]  # [B, obs_dim]

        # Flatten action chunk
        T = min(actions.shape[1], self.action_chunk_size)
        actions_flat = actions[:, :T].reshape(B, -1)  # [B, T*action_dim]

        # Pad if needed
        target_len = self.action_chunk_size * self.action_dim
        if actions_flat.shape[1] < target_len:
            pad = torch.zeros(
                B, target_len - actions_flat.shape[1], device=actions_flat.device
            )
            actions_flat = torch.cat([actions_flat, pad], dim=1)

        # Sample random timesteps
        t = torch.randint(0, self.n_diffusion_steps, (B,), device=state.device)

        # Add noise
        noisy_actions, noise = self._add_noise(actions_flat, t)

        # Predict noise
        pred_noise = self._predict_noise(noisy_actions, state_cond, t)

        # Loss
        loss = F.mse_loss(pred_noise, noise)

        return {"loss": loss}

    @torch.no_grad()
    def predict(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        state = obs["state"]
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim == 3:
            state = state[:, 0]  # [B, obs_dim]

        B = state.shape[0]
        flat_dim = self.action_chunk_size * self.action_dim

        # Start from pure noise
        x = torch.randn(B, flat_dim, device=state.device)

        # DDPM reverse process
        for i in reversed(range(self.n_diffusion_steps)):
            t = torch.full((B,), i, device=state.device, dtype=torch.long)
            pred_noise = self._predict_noise(x, state, t)

            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]

            # DDPM update
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise
            )

            if i > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta) * noise

        # Reshape to action chunk
        actions = x.reshape(B, self.action_chunk_size, self.action_dim)
        return actions

    def _get_config(self) -> dict:
        return {
            **super()._get_config(),
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "n_diffusion_steps": self.n_diffusion_steps,
            "noise_schedule": self.noise_schedule,
        }

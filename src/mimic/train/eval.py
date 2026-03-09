from __future__ import annotations

import numpy as np
import torch

from mimic.envs.base import MimicEnv
from mimic.train.policies.base import MimicPolicy


def evaluate_policy(
    policy: MimicPolicy,
    env: MimicEnv,
    n_episodes: int = 10,
    device: str = "cpu",
) -> dict:
    """Evaluate a policy in simulation."""
    policy.eval()
    successes = []
    returns = []

    for ep in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            state = torch.from_numpy(obs["state"]).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy.predict({"state": state})
            action_np = action.squeeze(0).cpu().numpy()

            # Use only first action if chunk
            if action_np.ndim > 1:
                action_np = action_np[0]

            obs, reward, done, info = env.step(action_np)
            total_reward += reward

        successes.append(info.get("is_success", False))
        returns.append(total_reward)

    policy.train()
    return {
        "success_rate": np.mean(successes),
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "n_episodes": n_episodes,
    }

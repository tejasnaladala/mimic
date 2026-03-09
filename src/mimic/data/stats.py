from __future__ import annotations

import numpy as np


def compute_dataset_stats(dataset) -> dict:
    """Compute per-feature statistics."""
    states = []
    actions = []
    rewards = []

    for ep in dataset._episodes:
        for frame in ep:
            states.append(frame["state"])
            actions.append(frame["action"])
            rewards.append(frame["reward"])

    stats = {}
    if states:
        states_arr = np.array(states)
        stats["state"] = {
            "mean": states_arr.mean(axis=0).tolist(),
            "std": states_arr.std(axis=0).tolist(),
            "min": states_arr.min(axis=0).tolist(),
            "max": states_arr.max(axis=0).tolist(),
        }
    if actions:
        actions_arr = np.array(actions)
        stats["action"] = {
            "mean": actions_arr.mean(axis=0).tolist(),
            "std": actions_arr.std(axis=0).tolist(),
            "min": actions_arr.min(axis=0).tolist(),
            "max": actions_arr.max(axis=0).tolist(),
        }
    if rewards:
        rewards_arr = np.array(rewards)
        stats["reward"] = {
            "mean": float(rewards_arr.mean()),
            "std": float(rewards_arr.std()),
            "min": float(rewards_arr.min()),
            "max": float(rewards_arr.max()),
        }

    return stats

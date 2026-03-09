from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset


class MimicTrainDataset(Dataset):
    """PyTorch dataset that loads episodes from Parquet files."""

    def __init__(
        self,
        dataset_path: str | Path,
        chunk_size: int = 10,
        normalize: bool = True,
    ):
        self.dataset_path = Path(dataset_path)
        self.chunk_size = chunk_size
        self.normalize = normalize
        self._frames: list[dict] = []
        self._stats: dict | None = None

        self._load_all_episodes()
        if normalize:
            self._load_stats()

    def _load_all_episodes(self):
        data_dir = self.dataset_path / "data" / "chunk-000"
        if not data_dir.exists():
            return
        for parquet_file in sorted(data_dir.glob("episode_*.parquet")):
            table = pq.read_table(parquet_file)
            ep_idx = table.column("episode_index")[0].as_py()
            for i in range(table.num_rows):
                frame = {
                    "state": np.array(
                        table.column("state")[i].as_py(), dtype=np.float32
                    ),
                    "action": np.array(
                        table.column("action")[i].as_py(), dtype=np.float32
                    ),
                    "episode_index": ep_idx,
                    "frame_index": table.column("frame_index")[i].as_py(),
                }
                self._frames.append(frame)

    def _load_stats(self):
        stats_path = self.dataset_path / "meta" / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                self._stats = json.load(f)

    def _normalize(self, arr: np.ndarray, key: str) -> np.ndarray:
        if self._stats and key in self._stats:
            mean = np.array(self._stats[key]["mean"], dtype=np.float32)
            std = np.array(self._stats[key]["std"], dtype=np.float32)
            std = np.where(std < 1e-6, 1.0, std)
            return (arr - mean) / std
        return arr

    def __len__(self) -> int:
        return max(0, len(self._frames) - self.chunk_size)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        frames = self._frames[idx : idx + self.chunk_size]

        states = np.stack([f["state"] for f in frames])
        actions = np.stack([f["action"] for f in frames])

        if self.normalize:
            states = self._normalize(states, "state")
            actions = self._normalize(actions, "action")

        return {
            "state": torch.from_numpy(states),
            "action": torch.from_numpy(actions),
        }


def create_dataloader(
    dataset_path: str | Path,
    batch_size: int = 32,
    chunk_size: int = 10,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    dataset = MimicTrainDataset(dataset_path, chunk_size=chunk_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

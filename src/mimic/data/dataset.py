from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class MimicDataset:
    """Dataset for storing robot demonstrations."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._episodes: list[list[dict]] = []  # list of episodes, each is list of frames
        self._current_episode: list[dict] = []
        self._metadata: dict[str, Any] = {}
        self._stats: dict | None = None

        if (self.path / "meta" / "info.json").exists():
            self._load_metadata()

    @classmethod
    def create(
        cls,
        path: str | Path,
        env_name: str,
        action_dim: int,
        state_dim: int,
        camera_names: list[str] | None = None,
        fps: int = 20,
    ) -> MimicDataset:
        """Create a new empty dataset."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "meta").mkdir(exist_ok=True)
        (path / "data").mkdir(exist_ok=True)
        (path / "videos").mkdir(exist_ok=True)

        dataset = cls(path)
        dataset._metadata = {
            "codebase_version": "v3.0",
            "robot_type": "mimic",
            "env_name": env_name,
            "fps": fps,
            "action_dim": action_dim,
            "state_dim": state_dim,
            "camera_names": camera_names or [],
            "num_episodes": 0,
            "num_frames": 0,
        }
        dataset._save_metadata()
        return dataset

    def add_frame(
        self,
        observation: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float = 0.0,
        done: bool = False,
        info: dict | None = None,
    ):
        """Add a frame to the current episode."""
        frame: dict[str, Any] = {
            "state": observation.get("state", np.array([])).tolist(),
            "joint_pos": observation.get("joint_pos", np.array([])).tolist(),
            "joint_vel": observation.get("joint_vel", np.array([])).tolist(),
            "action": action.tolist(),
            "reward": float(reward),
            "done": done,
            "timestamp": len(self._current_episode) / max(self._metadata.get("fps", 20), 1),
        }
        # Store images separately (will be encoded to video)
        frame["_images"] = {}
        for key, val in observation.items():
            if key.startswith("image."):
                cam_name = key.split(".", 1)[1]
                frame["_images"][cam_name] = val  # numpy array, not serialized

        self._current_episode.append(frame)

    def end_episode(self, task: str = "default") -> int:
        """Finalize the current episode and save it. Returns episode index."""
        if not self._current_episode:
            return -1

        episode_idx = len(self._episodes)
        episode = self._current_episode.copy()

        # Save numeric data as Parquet
        self._save_episode_parquet(episode_idx, episode, task)

        # Save video if images exist
        if episode[0].get("_images"):
            self._save_episode_videos(episode_idx, episode)

        self._episodes.append(episode)
        self._current_episode = []

        # Update metadata
        self._metadata["num_episodes"] = len(self._episodes)
        self._metadata["num_frames"] = sum(len(ep) for ep in self._episodes)
        self._save_metadata()

        return episode_idx

    def discard_episode(self):
        """Discard the current episode without saving."""
        self._current_episode = []

    def _save_episode_parquet(self, idx: int, episode: list[dict], task: str):
        """Save episode numeric data to Parquet."""
        chunk_dir = self.path / "data" / "chunk-000"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for i, frame in enumerate(episode):
            row = {
                "episode_index": idx,
                "frame_index": i,
                "timestamp": frame["timestamp"],
                "task": task,
                "state": frame["state"],
                "joint_pos": frame["joint_pos"],
                "joint_vel": frame["joint_vel"],
                "action": frame["action"],
                "reward": frame["reward"],
                "done": frame["done"],
            }
            rows.append(row)

        table = pa.table(
            {
                "episode_index": pa.array(
                    [r["episode_index"] for r in rows], type=pa.int32()
                ),
                "frame_index": pa.array([r["frame_index"] for r in rows], type=pa.int32()),
                "timestamp": pa.array([r["timestamp"] for r in rows], type=pa.float64()),
                "task": pa.array([r["task"] for r in rows], type=pa.string()),
                "state": pa.array([r["state"] for r in rows]),
                "joint_pos": pa.array([r["joint_pos"] for r in rows]),
                "joint_vel": pa.array([r["joint_vel"] for r in rows]),
                "action": pa.array([r["action"] for r in rows]),
                "reward": pa.array([r["reward"] for r in rows], type=pa.float64()),
                "done": pa.array([r["done"] for r in rows], type=pa.bool_()),
            }
        )

        pq.write_table(table, chunk_dir / f"episode_{idx:06d}.parquet")

    def _save_episode_videos(self, idx: int, episode: list[dict]):
        """Save episode camera images as MP4 videos using av."""
        import av

        camera_names = list(episode[0]["_images"].keys())
        fps = self._metadata.get("fps", 20)

        for cam_name in camera_names:
            video_dir = self.path / "videos" / "chunk-000" / cam_name
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"episode_{idx:06d}.mp4"

            # Get frame dimensions from first frame
            first_frame = episode[0]["_images"][cam_name]
            h, w = first_frame.shape[:2]

            container = av.open(str(video_path), mode="w")
            stream = container.add_stream("h264", rate=fps)
            stream.width = w
            stream.height = h
            stream.pix_fmt = "yuv420p"

            for frame_data in episode:
                img = frame_data["_images"].get(cam_name)
                if img is not None:
                    video_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                    for packet in stream.encode(video_frame):
                        container.mux(packet)

            # Flush
            for packet in stream.encode():
                container.mux(packet)
            container.close()

    def _save_metadata(self):
        meta_path = self.path / "meta" / "info.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def _load_metadata(self):
        with open(self.path / "meta" / "info.json") as f:
            self._metadata = json.load(f)

    def compute_stats(self) -> dict:
        """Compute per-feature statistics across all episodes."""
        from mimic.data.stats import compute_dataset_stats

        self._stats = compute_dataset_stats(self)
        # Save stats
        with open(self.path / "meta" / "stats.json", "w") as f:
            json.dump(self._stats, f, indent=2)
        return self._stats

    def load_episode(self, idx: int) -> pa.Table:
        """Load a single episode from Parquet."""
        parquet_path = self.path / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Episode {idx} not found at {parquet_path}")
        return pq.read_table(parquet_path)

    def __len__(self) -> int:
        return self._metadata.get("num_frames", 0)

    def __getitem__(self, idx: int) -> dict:
        """Get a single frame by global index."""
        # Find which episode and frame
        cumulative = 0
        for ep_idx, ep in enumerate(self._episodes):
            if idx < cumulative + len(ep):
                frame_idx = idx - cumulative
                frame = ep[frame_idx]
                return {
                    "state": np.array(frame["state"]),
                    "action": np.array(frame["action"]),
                    "reward": frame["reward"],
                    "done": frame["done"],
                    "timestamp": frame["timestamp"],
                    "episode_index": ep_idx,
                    "frame_index": frame_idx,
                }
            cumulative += len(ep)
        raise IndexError(f"Index {idx} out of range (dataset has {len(self)} frames)")

    @property
    def num_episodes(self) -> int:
        return self._metadata.get("num_episodes", 0)

    @property
    def metadata(self) -> dict:
        return self._metadata.copy()

    def delete(self):
        """Delete the entire dataset from disk."""
        if self.path.exists():
            shutil.rmtree(self.path)

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from mimic.data.dataset import MimicDataset
from mimic.envs.base import MimicEnv

logger = logging.getLogger(__name__)


class EpisodeRecorder:
    """Records teleoperation episodes into a MimicDataset."""

    def __init__(self, dataset: MimicDataset, env: MimicEnv):
        self.dataset = dataset
        self.env = env
        self._recording = False
        self._episode_count = 0
        self._frame_count = 0
        self._on_episode_end: Callable | None = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def current_frame_count(self) -> int:
        return self._frame_count

    def start_recording(self):
        if self._recording:
            logger.warning("Already recording")
            return
        self._recording = True
        self._frame_count = 0
        logger.info("Recording started")

    def stop_recording(self) -> int:
        if not self._recording:
            return -1
        self._recording = False
        episode_idx = self.dataset.end_episode()
        if episode_idx >= 0:
            self._episode_count += 1
            logger.info(f"Episode {episode_idx} saved ({self._frame_count} frames)")
            if self._on_episode_end:
                self._on_episode_end(episode_idx)
        return episode_idx

    def discard_recording(self):
        self._recording = False
        self._frame_count = 0
        self.dataset.discard_episode()
        logger.info("Episode discarded")

    def record_frame(
        self, obs: dict, action: np.ndarray, reward: float = 0.0, done: bool = False
    ):
        if not self._recording:
            return
        self.dataset.add_frame(obs, action, reward, done)
        self._frame_count += 1

    def on_episode_end(self, callback: Callable):
        self._on_episode_end = callback

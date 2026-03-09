"""Stack manipulation environment with a Franka Panda arm."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mimic.config.models import EnvConfig
from mimic.envs.base import MimicEnv
from mimic.envs.registry import register

SCENE_XML = Path(__file__).parent.parent / "assets" / "scenes" / "tabletop_stack.xml"

# Table surface parameters for randomizing positions
_TABLE_CENTER_X = 0.5
_TABLE_CENTER_Y = 0.0
_TABLE_Z = 0.445  # table top + half cube height
_SPAWN_RANGE_X = 0.12  # +/- from table center
_SPAWN_RANGE_Y = 0.15  # +/- from table center
_MIN_CUBE_DIST = 0.10  # minimum distance between the two cubes

# Freejoint qpos offsets (7 arm joints + 2 gripper joints = 9 qpos before cubes)
# Red cube freejoint: 7 qpos values (x, y, z, qw, qx, qy, qz) starting at index 9
_RED_CUBE_QPOS_ADDR = 9
# Blue cube freejoint: 7 qpos values starting at index 16
_BLUE_CUBE_QPOS_ADDR = 16

# Freejoint qvel offsets (7 arm joints + 2 gripper joints = 9 qvel before cubes)
# Red cube: 6 DOF (3 translational + 3 rotational) starting at index 9
_RED_CUBE_QVEL_ADDR = 9
# Blue cube: 6 DOF starting at index 15
_BLUE_CUBE_QVEL_ADDR = 15

# Stacking offset: red cube should be placed on top of blue cube
_STACK_OFFSET = np.array([0.0, 0.0, 0.05])

# Success threshold (meters)
_SUCCESS_THRESHOLD = 0.05


@register("stack")
class StackEnv(MimicEnv):
    """Pick up a red cube and stack it on top of a blue cube.

    The robot must pick up a red cube from the table surface and
    place it on top of a blue cube that sits on the table.

    Observation keys:
        state: full qpos + qvel
        joint_pos: joint positions
        joint_vel: joint velocities
        image.front: front camera RGB image (240x320x3)
        image.wrist: wrist camera RGB image (240x320x3)

    Action space:
        9-dim: 7 arm joint position targets + 2 gripper finger targets

    Reward:
        Negative Euclidean distance from red cube center to
        (blue cube center + [0, 0, 0.05]).

    Success:
        Red cube is within 0.05m of the stacking position AND
        red cube z > blue cube z.
    """

    def __init__(self, config: EnvConfig):
        super().__init__(config, SCENE_XML)
        self._rng = np.random.RandomState(42)

    def _reset_task(self):
        """Randomize red and blue cube positions on the table."""
        # Randomize red cube position
        red_x = _TABLE_CENTER_X + self._rng.uniform(-_SPAWN_RANGE_X, _SPAWN_RANGE_X)
        red_y = _TABLE_CENTER_Y + self._rng.uniform(-_SPAWN_RANGE_Y, _SPAWN_RANGE_Y)

        # Randomize blue cube position ensuring minimum distance from red cube
        for _ in range(100):
            blue_x = _TABLE_CENTER_X + self._rng.uniform(-_SPAWN_RANGE_X, _SPAWN_RANGE_X)
            blue_y = _TABLE_CENTER_Y + self._rng.uniform(-_SPAWN_RANGE_Y, _SPAWN_RANGE_Y)
            dist = np.sqrt((blue_x - red_x) ** 2 + (blue_y - red_y) ** 2)
            if dist >= _MIN_CUBE_DIST:
                break

        # Set red cube position via freejoint qpos (x, y, z, qw, qx, qy, qz)
        self.data.qpos[_RED_CUBE_QPOS_ADDR : _RED_CUBE_QPOS_ADDR + 3] = [
            red_x,
            red_y,
            _TABLE_Z,
        ]
        self.data.qpos[_RED_CUBE_QPOS_ADDR + 3 : _RED_CUBE_QPOS_ADDR + 7] = [
            1.0,
            0.0,
            0.0,
            0.0,
        ]

        # Set blue cube position via freejoint qpos
        self.data.qpos[_BLUE_CUBE_QPOS_ADDR : _BLUE_CUBE_QPOS_ADDR + 3] = [
            blue_x,
            blue_y,
            _TABLE_Z,
        ]
        self.data.qpos[_BLUE_CUBE_QPOS_ADDR + 3 : _BLUE_CUBE_QPOS_ADDR + 7] = [
            1.0,
            0.0,
            0.0,
            0.0,
        ]

        # Zero out cube velocities
        self.data.qvel[_RED_CUBE_QVEL_ADDR : _RED_CUBE_QVEL_ADDR + 6] = 0.0
        self.data.qvel[_BLUE_CUBE_QVEL_ADDR : _BLUE_CUBE_QVEL_ADDR + 6] = 0.0

    def _get_red_cube_pos(self) -> np.ndarray:
        """Get current red cube center position from qpos."""
        return self.data.qpos[_RED_CUBE_QPOS_ADDR : _RED_CUBE_QPOS_ADDR + 3].copy()

    def _get_blue_cube_pos(self) -> np.ndarray:
        """Get current blue cube center position from qpos."""
        return self.data.qpos[_BLUE_CUBE_QPOS_ADDR : _BLUE_CUBE_QPOS_ADDR + 3].copy()

    def _get_stack_target(self) -> np.ndarray:
        """Get the target stacking position (above blue cube)."""
        return self._get_blue_cube_pos() + _STACK_OFFSET

    def _compute_reward(self) -> float:
        """Dense reward: negative distance from red cube to stack target."""
        red_pos = self._get_red_cube_pos()
        target_pos = self._get_stack_target()
        dist = np.linalg.norm(red_pos - target_pos)
        return -float(dist)

    def _is_success(self) -> bool:
        """Success if red cube is near stack target AND above blue cube."""
        red_pos = self._get_red_cube_pos()
        blue_pos = self._get_blue_cube_pos()
        target_pos = self._get_stack_target()
        dist = np.linalg.norm(red_pos - target_pos)
        return bool(dist < _SUCCESS_THRESHOLD and red_pos[2] > blue_pos[2])

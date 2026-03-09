"""Tests for the pick-place manipulation environment."""

import numpy as np

import mimic.envs.tasks  # noqa: F401
from mimic.envs.registry import make


class TestPickPlace:
    def test_creates(self):
        env = make("pick-place")
        assert env is not None
        env.close()

    def test_reset_returns_obs(self):
        env = make("pick-place")
        obs = env.reset()
        assert "state" in obs
        assert "joint_pos" in obs
        assert "joint_vel" in obs
        assert "image.front" in obs
        assert "image.wrist" in obs
        assert obs["image.front"].shape == (240, 320, 3)
        assert obs["image.front"].dtype == np.uint8
        assert obs["image.wrist"].shape == (240, 320, 3)
        assert obs["image.wrist"].dtype == np.uint8
        env.close()

    def test_step_returns_tuple(self):
        env = make("pick-place")
        env.reset()
        action = np.zeros(env.action_dim)
        obs, reward, done, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "is_success" in info
        env.close()

    def test_action_dim(self):
        env = make("pick-place")
        # Panda has 7 arm joints + 2 gripper fingers = 9
        assert env.action_dim == 9
        assert env.action_dim >= 7
        env.close()

    def test_multiple_steps(self):
        env = make("pick-place")
        env.reset()
        for _ in range(10):
            action = np.zeros(env.action_dim)
            obs, reward, done, info = env.step(action)
            assert reward <= 0.0  # reward is negative distance
        env.close()

    def test_reset_randomizes(self):
        env = make("pick-place")
        obs1 = env.reset()
        obs2 = env.reset()
        # Cube positions in the state should differ due to randomization
        # Cube freejoint qpos starts at index 9 in joint_pos
        cube_pos1 = obs1["joint_pos"][9:12]
        cube_pos2 = obs2["joint_pos"][9:12]
        # With high probability, random positions will differ
        assert not np.allclose(cube_pos1, cube_pos2), (
            "Cube positions should be randomized across resets"
        )
        env.close()

    def test_reward_is_negative_distance(self):
        env = make("pick-place")
        env.reset()
        action = np.zeros(env.action_dim)
        _, reward, _, _ = env.step(action)
        # Reward should be non-positive (negative distance)
        assert reward <= 0.0
        env.close()

    def test_state_dim(self):
        env = make("pick-place")
        obs = env.reset()
        # state = qpos (16) + qvel (15) = 31
        assert obs["state"].shape[0] == env.state_dim
        assert obs["joint_pos"].shape[0] == 16  # 7 arm + 2 gripper + 7 cube freejoint
        assert obs["joint_vel"].shape[0] == 15  # 7 arm + 2 gripper + 6 cube freejoint
        env.close()

    def test_env_in_registry(self):
        from mimic.envs.registry import list_envs

        envs = list_envs()
        assert "pick-place" in envs

    def test_close_is_safe(self):
        env = make("pick-place")
        env.reset()
        env.close()
        # Closing again should not raise
        env.close()

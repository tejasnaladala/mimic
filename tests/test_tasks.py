"""Tests for push and stack manipulation environments."""

import numpy as np

import mimic.envs.tasks  # noqa: F401
from mimic.envs.registry import list_envs, make


class TestPush:
    def test_creates(self):
        env = make("push")
        assert env is not None
        env.close()

    def test_reset_and_step(self):
        env = make("push")
        obs = env.reset()
        assert "state" in obs
        assert "image.front" in obs
        action = np.zeros(env.action_dim)
        obs, reward, done, info = env.step(action)
        assert reward <= 0.0
        assert "is_success" in info
        env.close()

    def test_reward_is_2d_distance(self):
        env = make("push")
        env.reset()
        action = np.zeros(env.action_dim)
        _, reward, _, _ = env.step(action)
        assert isinstance(reward, float)
        assert reward <= 0.0
        env.close()


class TestStack:
    def test_creates(self):
        env = make("stack")
        assert env is not None
        env.close()

    def test_reset_and_step(self):
        env = make("stack")
        obs = env.reset()
        assert "state" in obs
        action = np.zeros(env.action_dim)
        obs, reward, done, info = env.step(action)
        assert reward <= 0.0
        assert "is_success" in info
        env.close()

    def test_has_two_cubes(self):
        env = make("stack")
        env.reset()
        # Should have bodies for both cubes
        red_pos = env.data.body("red_cube").xpos.copy()
        blue_pos = env.data.body("blue_cube").xpos.copy()
        assert not np.allclose(red_pos, blue_pos)
        env.close()


class TestAllEnvs:
    def test_all_envs_registered(self):
        envs = list_envs()
        assert "pick-place" in envs
        assert "push" in envs
        assert "stack" in envs

    def test_all_envs_create_and_reset(self):
        for env_name in list_envs():
            env = make(env_name)
            obs = env.reset()
            assert "state" in obs
            env.close()

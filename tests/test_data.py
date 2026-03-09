import numpy as np

from mimic.data.dataset import MimicDataset
from mimic.data.recorder import EpisodeRecorder


class TestMimicDataset:
    def test_create(self, tmp_path):
        ds = MimicDataset.create(
            tmp_path / "test_ds", env_name="pick-place", action_dim=9, state_dim=18
        )
        assert ds.num_episodes == 0
        assert len(ds) == 0
        assert (tmp_path / "test_ds" / "meta" / "info.json").exists()

    def test_add_frames_and_end_episode(self, tmp_path):
        ds = MimicDataset.create(tmp_path / "test_ds", env_name="test", action_dim=2, state_dim=4)
        for i in range(10):
            obs = {
                "state": np.random.randn(4),
                "joint_pos": np.random.randn(2),
                "joint_vel": np.random.randn(2),
            }
            action = np.random.randn(2)
            ds.add_frame(obs, action, reward=-0.5)
        idx = ds.end_episode()
        assert idx == 0
        assert ds.num_episodes == 1
        assert len(ds) == 10

    def test_multiple_episodes(self, tmp_path):
        ds = MimicDataset.create(tmp_path / "ds", env_name="test", action_dim=2, state_dim=4)
        for ep in range(3):
            for i in range(5):
                obs = {
                    "state": np.zeros(4),
                    "joint_pos": np.zeros(2),
                    "joint_vel": np.zeros(2),
                }
                ds.add_frame(obs, np.zeros(2))
            ds.end_episode()
        assert ds.num_episodes == 3
        assert len(ds) == 15

    def test_discard_episode(self, tmp_path):
        ds = MimicDataset.create(tmp_path / "ds", env_name="test", action_dim=2, state_dim=4)
        for i in range(5):
            obs = {"state": np.zeros(4), "joint_pos": np.zeros(2), "joint_vel": np.zeros(2)}
            ds.add_frame(obs, np.zeros(2))
        ds.discard_episode()
        assert ds.num_episodes == 0

    def test_getitem(self, tmp_path):
        ds = MimicDataset.create(tmp_path / "ds", env_name="test", action_dim=2, state_dim=4)
        for i in range(5):
            obs = {
                "state": np.ones(4) * i,
                "joint_pos": np.zeros(2),
                "joint_vel": np.zeros(2),
            }
            ds.add_frame(obs, np.ones(2) * i, reward=-float(i))
        ds.end_episode()
        frame = ds[2]
        assert frame["frame_index"] == 2
        assert frame["episode_index"] == 0
        np.testing.assert_array_almost_equal(frame["state"], np.ones(4) * 2)

    def test_parquet_saved(self, tmp_path):
        ds = MimicDataset.create(tmp_path / "ds", env_name="test", action_dim=2, state_dim=4)
        obs = {"state": np.zeros(4), "joint_pos": np.zeros(2), "joint_vel": np.zeros(2)}
        ds.add_frame(obs, np.zeros(2))
        ds.end_episode()
        parquet_path = tmp_path / "ds" / "data" / "chunk-000" / "episode_000000.parquet"
        assert parquet_path.exists()

    def test_load_episode(self, tmp_path):
        ds = MimicDataset.create(tmp_path / "ds", env_name="test", action_dim=2, state_dim=4)
        for i in range(3):
            obs = {"state": np.zeros(4), "joint_pos": np.zeros(2), "joint_vel": np.zeros(2)}
            ds.add_frame(obs, np.zeros(2))
        ds.end_episode()
        table = ds.load_episode(0)
        assert table.num_rows == 3

    def test_video_saved_with_images(self, tmp_path):
        ds = MimicDataset.create(
            tmp_path / "ds",
            env_name="test",
            action_dim=2,
            state_dim=4,
            camera_names=["front"],
            fps=10,
        )
        for i in range(5):
            obs = {
                "state": np.zeros(4),
                "joint_pos": np.zeros(2),
                "joint_vel": np.zeros(2),
                "image.front": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            }
            ds.add_frame(obs, np.zeros(2))
        ds.end_episode()
        video_path = tmp_path / "ds" / "videos" / "chunk-000" / "front" / "episode_000000.mp4"
        assert video_path.exists()

    def test_metadata_persistence(self, tmp_path):
        ds = MimicDataset.create(tmp_path / "ds", env_name="test", action_dim=2, state_dim=4)
        obs = {"state": np.zeros(4), "joint_pos": np.zeros(2), "joint_vel": np.zeros(2)}
        ds.add_frame(obs, np.zeros(2))
        ds.end_episode()
        # Reload
        ds2 = MimicDataset(tmp_path / "ds")
        assert ds2.metadata["num_episodes"] == 1

    def test_delete(self, tmp_path):
        ds = MimicDataset.create(tmp_path / "ds", env_name="test", action_dim=2, state_dim=4)
        assert (tmp_path / "ds").exists()
        ds.delete()
        assert not (tmp_path / "ds").exists()


class TestStats:
    def test_compute_stats(self, tmp_path):
        ds = MimicDataset.create(tmp_path / "ds", env_name="test", action_dim=2, state_dim=4)
        for i in range(10):
            obs = {
                "state": np.random.randn(4),
                "joint_pos": np.zeros(2),
                "joint_vel": np.zeros(2),
            }
            ds.add_frame(obs, np.random.randn(2), reward=np.random.randn())
        ds.end_episode()
        stats = ds.compute_stats()
        assert "state" in stats
        assert "action" in stats
        assert "reward" in stats
        assert "mean" in stats["state"]
        assert "std" in stats["state"]
        assert (tmp_path / "ds" / "meta" / "stats.json").exists()


class TestRecorder:
    def test_creates(self, tmp_path):
        import mimic.envs.tasks  # noqa: F401
        from mimic.envs.registry import make

        ds = MimicDataset.create(
            tmp_path / "ds", env_name="pick-place", action_dim=9, state_dim=18
        )
        env = make("pick-place")
        recorder = EpisodeRecorder(ds, env)
        assert not recorder.is_recording
        env.close()

    def test_record_episode(self, tmp_path):
        import mimic.envs.tasks  # noqa: F401
        from mimic.envs.registry import make

        env = make("pick-place")
        ds = MimicDataset.create(
            tmp_path / "ds",
            env_name="pick-place",
            action_dim=env.action_dim,
            state_dim=env.state_dim,
        )
        recorder = EpisodeRecorder(ds, env)

        recorder.start_recording()
        assert recorder.is_recording
        obs = env.reset()
        for _ in range(5):
            action = np.zeros(env.action_dim)
            recorder.record_frame(obs, action, reward=-1.0)
            obs, _, _, _ = env.step(action)

        ep_idx = recorder.stop_recording()
        assert ep_idx == 0
        assert recorder.episode_count == 1
        assert ds.num_episodes == 1
        env.close()

    def test_discard(self, tmp_path):
        import mimic.envs.tasks  # noqa: F401
        from mimic.envs.registry import make

        env = make("pick-place")
        ds = MimicDataset.create(
            tmp_path / "ds",
            env_name="pick-place",
            action_dim=env.action_dim,
            state_dim=env.state_dim,
        )
        recorder = EpisodeRecorder(ds, env)
        recorder.start_recording()
        obs = env.reset()
        recorder.record_frame(obs, np.zeros(env.action_dim))
        recorder.discard_recording()
        assert not recorder.is_recording
        assert ds.num_episodes == 0
        env.close()

    def test_not_recording_ignores_frames(self, tmp_path):
        import mimic.envs.tasks  # noqa: F401
        from mimic.envs.registry import make

        env = make("pick-place")
        ds = MimicDataset.create(
            tmp_path / "ds",
            env_name="pick-place",
            action_dim=env.action_dim,
            state_dim=env.state_dim,
        )
        recorder = EpisodeRecorder(ds, env)
        obs = env.reset()
        recorder.record_frame(obs, np.zeros(env.action_dim))  # not recording
        assert recorder.current_frame_count == 0
        env.close()

import numpy as np
import torch

from mimic.train.policies.act import ACTPolicy


class TestACTPolicy:
    def test_creates(self):
        policy = ACTPolicy(obs_dim=18, action_dim=9, action_chunk_size=10)
        assert policy.obs_dim == 18
        assert policy.action_dim == 9

    def test_forward(self):
        policy = ACTPolicy(
            obs_dim=18, action_dim=9, action_chunk_size=10, hidden_dim=64, n_layers=2
        )
        batch = {
            "state": torch.randn(4, 10, 18),
            "action": torch.randn(4, 10, 9),
        }
        output = policy.forward(batch)
        assert "loss" in output
        assert "recon_loss" in output
        assert "kl_loss" in output
        assert output["loss"].requires_grad

    def test_predict(self):
        policy = ACTPolicy(
            obs_dim=18, action_dim=9, action_chunk_size=10, hidden_dim=64, n_layers=2
        )
        policy.eval()
        obs = {"state": torch.randn(1, 18)}
        with torch.no_grad():
            actions = policy.predict(obs)
        assert actions.shape == (1, 10, 9)

    def test_save_load(self, tmp_path):
        policy = ACTPolicy(
            obs_dim=18, action_dim=9, action_chunk_size=10, hidden_dim=64, n_layers=2
        )
        path = str(tmp_path / "test_policy.pt")
        policy.save(path)
        loaded = ACTPolicy.load(path)
        assert loaded.obs_dim == 18
        assert loaded.action_dim == 9

    def test_backward(self):
        policy = ACTPolicy(
            obs_dim=4, action_dim=2, action_chunk_size=5, hidden_dim=32, n_layers=1
        )
        optimizer = policy.get_optimizer(lr=1e-3)
        batch = {"state": torch.randn(2, 5, 4), "action": torch.randn(2, 5, 2)}
        output = policy.forward(batch)
        optimizer.zero_grad()
        output["loss"].backward()
        optimizer.step()

    def test_predict_single_obs(self):
        policy = ACTPolicy(
            obs_dim=4, action_dim=2, action_chunk_size=5, hidden_dim=32, n_layers=1
        )
        policy.eval()
        obs = {"state": torch.randn(4)}  # single observation, no batch dim
        with torch.no_grad():
            actions = policy.predict(obs)
        assert actions.shape == (1, 5, 2)


class TestTrainer:
    def test_trainer_creates(self, tmp_path):
        from mimic.config.models import TrainConfig
        from mimic.data.dataset import MimicDataset
        from mimic.train.trainer import MimicTrainer

        # Create a small dataset
        ds = MimicDataset.create(
            tmp_path / "ds", env_name="test", action_dim=2, state_dim=4
        )
        for ep in range(2):
            for i in range(20):
                obs = {
                    "state": np.random.randn(4),
                    "joint_pos": np.zeros(2),
                    "joint_vel": np.zeros(2),
                }
                ds.add_frame(obs, np.random.randn(2))
            ds.end_episode()
        ds.compute_stats()

        policy = ACTPolicy(
            obs_dim=4, action_dim=2, action_chunk_size=5, hidden_dim=32, n_layers=1
        )
        config = TrainConfig(batch_size=4, lr=1e-3, steps=10, device="cpu")
        trainer = MimicTrainer(
            policy, config, tmp_path / "ds", output_dir=tmp_path / "out"
        )
        assert trainer.current_step == 0

    def test_trainer_runs(self, tmp_path):
        from mimic.config.models import TrainConfig
        from mimic.data.dataset import MimicDataset
        from mimic.train.trainer import MimicTrainer

        ds = MimicDataset.create(
            tmp_path / "ds", env_name="test", action_dim=2, state_dim=4
        )
        for ep in range(2):
            for i in range(20):
                obs = {
                    "state": np.random.randn(4),
                    "joint_pos": np.zeros(2),
                    "joint_vel": np.zeros(2),
                }
                ds.add_frame(obs, np.random.randn(2))
            ds.end_episode()
        ds.compute_stats()

        policy = ACTPolicy(
            obs_dim=4, action_dim=2, action_chunk_size=5, hidden_dim=32, n_layers=1
        )
        config = TrainConfig(
            batch_size=4, lr=1e-3, steps=10, save_every=0, device="cpu"
        )
        trainer = MimicTrainer(
            policy, config, tmp_path / "ds", output_dir=tmp_path / "out"
        )
        trainer.train(steps=10)
        assert trainer.current_step == 10
        assert trainer.recent_loss < float("inf")


class TestEval:
    def test_evaluate_policy(self):
        import mimic.envs.tasks  # noqa: F401
        from mimic.envs.registry import make
        from mimic.train.eval import evaluate_policy

        env = make("pick-place")
        policy = ACTPolicy(
            obs_dim=env.state_dim,
            action_dim=env.action_dim,
            action_chunk_size=5,
            hidden_dim=32,
            n_layers=1,
        )
        results = evaluate_policy(policy, env, n_episodes=2)
        assert "success_rate" in results
        assert "mean_return" in results
        assert results["n_episodes"] == 2
        env.close()

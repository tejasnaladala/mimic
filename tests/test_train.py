import numpy as np
import pytest
import torch

from mimic.data.dataset import MimicDataset
from mimic.train.dataloader import MimicTrainDataset
from mimic.train.policies.act import ACTPolicy


class _MaliciousCheckpoint:
    def __init__(self, marker_path: str):
        self.marker_path = marker_path

    def __reduce__(self):
        code = f"open({self.marker_path!r}, 'w').write('executed')"
        return exec, (code,)


def test_training_windows_never_cross_episode_boundaries(tmp_path) -> None:
    dataset_path = tmp_path / "dataset"
    recorded = MimicDataset.create(dataset_path, env_name="test", action_dim=1, state_dim=1)
    for value in (0.0, 1.0):
        recorded.add_frame({"state": np.array([value])}, np.array([value]))
    recorded.end_episode()
    for value in (100.0, 101.0, 102.0, 103.0):
        recorded.add_frame({"state": np.array([value])}, np.array([value]))
    recorded.end_episode()

    dataset = MimicTrainDataset(dataset_path, chunk_size=3, normalize=False)

    assert len(dataset) == 6
    for item in dataset:
        valid_states = item["state"][~item["is_pad"], 0]
        if item["episode_index"].item() == 0:
            assert torch.all(valid_states < 50)
        else:
            assert torch.all(valid_states >= 100)

    final_full_window = dataset[3]
    assert final_full_window["frame_index"].item() == 1
    assert final_full_window["state"][:, 0].tolist() == [101.0, 102.0, 103.0]
    assert final_full_window["is_pad"].tolist() == [False, False, False]

    final_suffix = dataset[-1]
    assert final_suffix["frame_index"].item() == 3
    assert final_suffix["state"][:, 0].tolist() == [103.0, 0.0, 0.0]
    assert final_suffix["is_pad"].tolist() == [False, True, True]


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

    def test_padding_is_excluded_from_context_and_loss(self):
        policy = ACTPolicy(
            obs_dim=4,
            action_dim=2,
            action_chunk_size=5,
            hidden_dim=32,
            n_layers=1,
            dropout=0.0,
        )
        policy.eval()
        state = torch.randn(2, 5, 4)
        actions = torch.randn(2, 5, 2)
        is_pad = torch.tensor([[False, False, False, True, True]] * 2)
        altered = actions.clone()
        altered[:, 3:] = 1_000_000

        torch.manual_seed(123)
        original_loss = policy.forward(
            {"state": state, "action": actions, "is_pad": is_pad}
        )["loss"]
        torch.manual_seed(123)
        altered_loss = policy.forward(
            {"state": state, "action": altered, "is_pad": is_pad}
        )["loss"]

        assert torch.allclose(original_loss, altered_loss)

    def test_rejects_pickle_capable_checkpoint(self, tmp_path):
        marker = tmp_path / "executed.txt"
        checkpoint = tmp_path / "malicious.pt"
        torch.save(
            {"state_dict": _MaliciousCheckpoint(str(marker)), "config": {}},
            checkpoint,
        )

        with pytest.raises(RuntimeError, match="tensor-only"):
            ACTPolicy.load(str(checkpoint))
        assert not marker.exists()


class TestDiffusionPolicy:
    def test_creates(self):
        from mimic.train.policies.diffusion import DiffusionPolicy

        policy = DiffusionPolicy(
            obs_dim=18,
            action_dim=9,
            action_chunk_size=10,
            hidden_dim=64,
            n_layers=2,
            n_diffusion_steps=10,
        )
        assert policy.obs_dim == 18

    def test_forward(self):
        from mimic.train.policies.diffusion import DiffusionPolicy

        policy = DiffusionPolicy(
            obs_dim=4,
            action_dim=2,
            action_chunk_size=5,
            hidden_dim=32,
            n_layers=2,
            n_diffusion_steps=10,
        )
        batch = {"state": torch.randn(2, 5, 4), "action": torch.randn(2, 5, 2)}
        output = policy.forward(batch)
        assert "loss" in output
        assert output["loss"].requires_grad

    def test_predict(self):
        from mimic.train.policies.diffusion import DiffusionPolicy

        policy = DiffusionPolicy(
            obs_dim=4,
            action_dim=2,
            action_chunk_size=5,
            hidden_dim=32,
            n_layers=2,
            n_diffusion_steps=5,
        )
        policy.eval()
        obs = {"state": torch.randn(1, 4)}
        actions = policy.predict(obs)
        assert actions.shape == (1, 5, 2)

    def test_backward(self):
        from mimic.train.policies.diffusion import DiffusionPolicy

        policy = DiffusionPolicy(
            obs_dim=4,
            action_dim=2,
            action_chunk_size=5,
            hidden_dim=32,
            n_layers=2,
            n_diffusion_steps=10,
        )
        optimizer = policy.get_optimizer(lr=1e-3)
        batch = {"state": torch.randn(2, 5, 4), "action": torch.randn(2, 5, 2)}
        output = policy.forward(batch)
        optimizer.zero_grad()
        output["loss"].backward()
        optimizer.step()

    def test_save_load(self, tmp_path):
        from mimic.train.policies.diffusion import DiffusionPolicy

        policy = DiffusionPolicy(
            obs_dim=4,
            action_dim=2,
            action_chunk_size=5,
            hidden_dim=32,
            n_layers=2,
            n_diffusion_steps=10,
        )
        path = str(tmp_path / "diff_policy.pt")
        policy.save(path)
        loaded = DiffusionPolicy.load(path)
        assert loaded.obs_dim == 4
        assert loaded.n_diffusion_steps == 10

    def test_padding_is_excluded_from_input_and_loss(self):
        from mimic.train.policies.diffusion import DiffusionPolicy

        policy = DiffusionPolicy(
            obs_dim=4,
            action_dim=2,
            action_chunk_size=5,
            hidden_dim=32,
            n_layers=2,
            n_diffusion_steps=10,
        )
        policy.eval()
        state = torch.randn(2, 5, 4)
        actions = torch.randn(2, 5, 2)
        is_pad = torch.tensor([[False, False, False, True, True]] * 2)
        altered = actions.clone()
        altered[:, 3:] = 1_000_000

        torch.manual_seed(321)
        original_loss = policy.forward(
            {"state": state, "action": actions, "is_pad": is_pad}
        )["loss"]
        torch.manual_seed(321)
        altered_loss = policy.forward(
            {"state": state, "action": altered, "is_pad": is_pad}
        )["loss"]

        assert torch.allclose(original_loss, altered_loss)


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

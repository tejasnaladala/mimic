
import numpy as np
import pytest


def _onnx_available():
    """Check if ONNX export dependencies are properly installed."""
    try:
        import onnx  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


needs_onnx = pytest.mark.skipif(not _onnx_available(), reason="onnx package not available")


class TestONNXExport:
    @needs_onnx
    def test_export_act(self, tmp_path):
        from mimic.deploy.export import export_to_onnx
        from mimic.train.policies.act import ACTPolicy

        policy = ACTPolicy(
            obs_dim=4, action_dim=2, action_chunk_size=5, hidden_dim=32, n_layers=1
        )
        ckpt_path = tmp_path / "act.pt"
        policy.save(str(ckpt_path))

        onnx_path = tmp_path / "act.onnx"
        result = export_to_onnx(ckpt_path, onnx_path)
        assert result.exists()

    @needs_onnx
    def test_export_diffusion(self, tmp_path):
        from mimic.deploy.export import export_to_onnx
        from mimic.train.policies.diffusion import DiffusionPolicy

        policy = DiffusionPolicy(
            obs_dim=4,
            action_dim=2,
            action_chunk_size=5,
            hidden_dim=32,
            n_layers=2,
            n_diffusion_steps=5,
        )
        ckpt_path = tmp_path / "diff.pt"
        policy.save(str(ckpt_path))

        onnx_path = tmp_path / "diff.onnx"
        result = export_to_onnx(ckpt_path, onnx_path)
        assert result.exists()

    @needs_onnx
    def test_verify_onnx(self, tmp_path):
        from mimic.deploy.export import export_to_onnx, verify_onnx
        from mimic.train.policies.act import ACTPolicy

        policy = ACTPolicy(
            obs_dim=4, action_dim=2, action_chunk_size=5, hidden_dim=32, n_layers=1
        )
        ckpt_path = tmp_path / "act.pt"
        policy.save(str(ckpt_path))
        onnx_path = export_to_onnx(ckpt_path, tmp_path / "act.onnx")

        assert verify_onnx(onnx_path, obs_dim=4)


class TestInferenceServer:
    def test_torch_backend(self, tmp_path):
        from mimic.deploy.inference import InferenceServer
        from mimic.train.policies.act import ACTPolicy

        policy = ACTPolicy(
            obs_dim=4, action_dim=2, action_chunk_size=5, hidden_dim=32, n_layers=1
        )
        path = tmp_path / "model.pt"
        policy.save(str(path))

        server = InferenceServer(path, backend="torch")
        assert server.is_loaded
        action = server.predict(np.random.randn(4))
        assert action.shape == (2,)

    @needs_onnx
    def test_onnx_backend(self, tmp_path):
        from mimic.deploy.export import export_to_onnx
        from mimic.deploy.inference import InferenceServer
        from mimic.train.policies.act import ACTPolicy

        policy = ACTPolicy(
            obs_dim=4, action_dim=2, action_chunk_size=5, hidden_dim=32, n_layers=1
        )
        ckpt = tmp_path / "model.pt"
        policy.save(str(ckpt))
        onnx_path = export_to_onnx(ckpt, tmp_path / "model.onnx")

        server = InferenceServer(onnx_path, backend="onnx")
        assert server.is_loaded
        action = server.predict(np.random.randn(4))
        assert action.shape == (2,)

    def test_action_buffering(self, tmp_path):
        from mimic.deploy.inference import InferenceServer
        from mimic.train.policies.act import ACTPolicy

        policy = ACTPolicy(
            obs_dim=4, action_dim=2, action_chunk_size=5, hidden_dim=32, n_layers=1
        )
        path = tmp_path / "model.pt"
        policy.save(str(path))

        server = InferenceServer(path, backend="torch")
        state = np.random.randn(4)
        # First call predicts full chunk
        a1 = server.predict(state)
        # Subsequent calls return from buffer
        a2 = server.predict(state)
        assert a1.shape == a2.shape == (2,)
        # After chunk_size calls, buffer should reset
        for _ in range(3):
            server.predict(state)
        # This should trigger a new prediction
        a_new = server.predict(state)
        assert a_new.shape == (2,)

    def test_reset_clears_buffer(self, tmp_path):
        from mimic.deploy.inference import InferenceServer
        from mimic.train.policies.act import ACTPolicy

        policy = ACTPolicy(
            obs_dim=4, action_dim=2, action_chunk_size=5, hidden_dim=32, n_layers=1
        )
        path = tmp_path / "model.pt"
        policy.save(str(path))

        server = InferenceServer(path, backend="torch")
        server.predict(np.random.randn(4))
        server.reset()
        assert server._action_buffer is None

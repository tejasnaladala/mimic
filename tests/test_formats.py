import numpy as np

from mimic.data.dataset import MimicDataset
from mimic.data.formats import export_to_lerobot, export_to_rlds


class TestExportFormats:
    def _create_test_dataset(self, path):
        ds = MimicDataset.create(path, env_name="test", action_dim=2, state_dim=4)
        for ep in range(2):
            for i in range(5):
                obs = {
                    "state": np.random.randn(4),
                    "joint_pos": np.zeros(2),
                    "joint_vel": np.zeros(2),
                }
                ds.add_frame(obs, np.random.randn(2), reward=-0.5)
            ds.end_episode()
        return ds

    def test_export_lerobot(self, tmp_path):
        self._create_test_dataset(tmp_path / "source")
        export_to_lerobot(tmp_path / "source", tmp_path / "lerobot_out")
        assert (tmp_path / "lerobot_out" / "meta" / "info.json").exists()
        assert (tmp_path / "lerobot_out" / "data" / "chunk-000").exists()

    def test_export_rlds(self, tmp_path):
        self._create_test_dataset(tmp_path / "source")
        export_to_rlds(tmp_path / "source", tmp_path / "rlds_out")
        assert (tmp_path / "rlds_out" / "episodes.json").exists()
        assert (tmp_path / "rlds_out" / "episode_000000" / "state.npy").exists()
        assert (tmp_path / "rlds_out" / "episode_000000" / "action.npy").exists()

    def test_export_hdf5_skipped_without_h5py(self, tmp_path):
        """Test that HDF5 export raises ImportError if h5py not installed."""
        self._create_test_dataset(tmp_path / "source")
        try:
            from mimic.data.formats import export_to_hdf5

            export_to_hdf5(tmp_path / "source", tmp_path / "out.hdf5")
            # If h5py is installed, check the file exists
            assert (tmp_path / "out.hdf5").exists()
        except ImportError:
            pass  # Expected if h5py not installed

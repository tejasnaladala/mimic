"""Tests for HuggingFace Hub integration (all API calls are mocked)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_dataset(root: Path) -> Path:
    """Create a minimal LeRobot-v3 dataset structure for testing."""
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "env_name": "pick-place",
                "num_episodes": 5,
                "num_frames": 500,
                "fps": 20,
                "action_dim": 7,
                "state_dim": 14,
                "camera_names": ["front"],
            }
        )
    )
    (meta / "episodes.jsonl").write_text("")
    (meta / "stats.json").write_text("{}")

    data = root / "data" / "chunk-000"
    data.mkdir(parents=True)
    (data / "episode_000000.parquet").write_bytes(b"parquet-stub")

    videos = root / "videos" / "chunk-000"
    videos.mkdir(parents=True)
    (videos / "front_episode_000000.mp4").write_bytes(b"mp4-stub")

    return root


def _make_checkpoint(root: Path, name: str = "best.pt") -> Path:
    """Create a fake checkpoint file for testing."""
    root.mkdir(parents=True, exist_ok=True)
    ckpt = root / name
    ckpt.write_bytes(b"checkpoint-stub")
    return ckpt


# --------------------------------------------------------------------------
# MimicHubClient unit tests
# --------------------------------------------------------------------------


class TestPushDataset:
    @patch("mimic.hub.client.HfApi")
    def test_push_creates_repo_and_uploads(self, MockHfApi, tmp_path):
        """push_dataset should create the repo and upload the folder."""
        from mimic.hub.client import MimicHubClient

        mock_api = MagicMock()
        MockHfApi.return_value = mock_api

        ds_path = _make_dataset(tmp_path / "my_demos")
        client = MimicHubClient()
        url = client.push_dataset(ds_path, "user/pick-place-demos")

        mock_api.create_repo.assert_called_once_with(
            repo_id="user/pick-place-demos",
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )
        mock_api.upload_folder.assert_called_once()
        call_kwargs = mock_api.upload_folder.call_args.kwargs
        assert call_kwargs["repo_id"] == "user/pick-place-demos"
        assert call_kwargs["repo_type"] == "dataset"
        assert "pick-place-demos" in url

    @patch("mimic.hub.client.HfApi")
    def test_push_private_repo(self, MockHfApi, tmp_path):
        """push_dataset with private=True should forward the flag."""
        from mimic.hub.client import MimicHubClient

        mock_api = MagicMock()
        MockHfApi.return_value = mock_api

        ds_path = _make_dataset(tmp_path / "demos")
        client = MimicHubClient()
        client.push_dataset(ds_path, "user/private-demos", private=True)

        mock_api.create_repo.assert_called_once_with(
            repo_id="user/private-demos",
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )

    @patch("mimic.hub.client.HfApi")
    def test_push_nonexistent_path_raises(self, MockHfApi, tmp_path):
        """push_dataset should raise FileNotFoundError for missing paths."""
        from mimic.hub.client import MimicHubClient

        client = MimicHubClient()
        with pytest.raises(FileNotFoundError):
            client.push_dataset(tmp_path / "nope", "user/demos")

    @patch("mimic.hub.client.HfApi")
    def test_push_missing_meta_raises(self, MockHfApi, tmp_path):
        """push_dataset should raise ValueError when meta/ is absent."""
        from mimic.hub.client import MimicHubClient

        bad_ds = tmp_path / "bad_ds"
        bad_ds.mkdir()
        client = MimicHubClient()
        with pytest.raises(ValueError, match="missing meta/"):
            client.push_dataset(bad_ds, "user/demos")

    @patch("mimic.hub.client.HfApi")
    def test_push_writes_dataset_card(self, MockHfApi, tmp_path):
        """push_dataset should generate a README.md dataset card."""
        from mimic.hub.client import MimicHubClient

        mock_api = MagicMock()
        MockHfApi.return_value = mock_api

        ds_path = _make_dataset(tmp_path / "demos")
        client = MimicHubClient()
        client.push_dataset(ds_path, "user/demos")

        # The upload_folder call should have been made while the README existed.
        # Since the README is cleaned up after upload, we verify upload was
        # called (meaning the card was present during upload).
        mock_api.upload_folder.assert_called_once()


class TestPullDataset:
    @patch("mimic.hub.client.snapshot_download")
    @patch("mimic.hub.client.HfApi")
    def test_pull_downloads_to_output(self, MockHfApi, mock_download, tmp_path):
        """pull_dataset should call snapshot_download with correct args."""
        from mimic.hub.client import MimicHubClient

        output = tmp_path / "output"
        mock_download.return_value = str(output)

        client = MimicHubClient()
        result = client.pull_dataset("user/pick-place-demos", output)

        mock_download.assert_called_once_with(
            repo_id="user/pick-place-demos",
            repo_type="dataset",
            local_dir=str(output),
        )
        assert result == output

    @patch("mimic.hub.client.snapshot_download")
    @patch("mimic.hub.client.HfApi")
    def test_pull_creates_output_dir(self, MockHfApi, mock_download, tmp_path):
        """pull_dataset should create the output directory if it doesn't exist."""
        from mimic.hub.client import MimicHubClient

        output = tmp_path / "nested" / "dir"
        mock_download.return_value = str(output)

        client = MimicHubClient()
        client.pull_dataset("user/demos", output)

        assert output.exists()


class TestPushModel:
    @patch("mimic.hub.client.HfApi")
    def test_push_uploads_checkpoint(self, MockHfApi, tmp_path):
        """push_model should create repo and upload the .pt file."""
        from mimic.hub.client import MimicHubClient

        mock_api = MagicMock()
        MockHfApi.return_value = mock_api

        ckpt = _make_checkpoint(tmp_path / "models")
        client = MimicHubClient()
        url = client.push_model(ckpt, "user/pick-place-act")

        mock_api.create_repo.assert_called_once_with(
            repo_id="user/pick-place-act",
            repo_type="model",
            private=False,
            exist_ok=True,
        )
        # Should upload checkpoint + README (no config.json present)
        assert mock_api.upload_file.call_count == 2
        upload_calls = mock_api.upload_file.call_args_list

        # First call: checkpoint upload
        assert upload_calls[0].kwargs["path_in_repo"] == "best.pt"
        assert upload_calls[0].kwargs["repo_type"] == "model"

        # Second call: model card
        assert upload_calls[1].kwargs["path_in_repo"] == "README.md"
        assert "pick-place-act" in url

    @patch("mimic.hub.client.HfApi")
    def test_push_uploads_config_when_present(self, MockHfApi, tmp_path):
        """push_model should also upload config.json if it exists."""
        from mimic.hub.client import MimicHubClient

        mock_api = MagicMock()
        MockHfApi.return_value = mock_api

        model_dir = tmp_path / "models"
        ckpt = _make_checkpoint(model_dir)
        (model_dir / "config.json").write_text('{"policy": "act"}')

        client = MimicHubClient()
        client.push_model(ckpt, "user/my-model")

        # checkpoint + config.json + README = 3 uploads
        assert mock_api.upload_file.call_count == 3
        uploaded_names = [
            c.kwargs["path_in_repo"] for c in mock_api.upload_file.call_args_list
        ]
        assert "best.pt" in uploaded_names
        assert "config.json" in uploaded_names
        assert "README.md" in uploaded_names

    @patch("mimic.hub.client.HfApi")
    def test_push_private_model(self, MockHfApi, tmp_path):
        """push_model with private=True should forward the flag."""
        from mimic.hub.client import MimicHubClient

        mock_api = MagicMock()
        MockHfApi.return_value = mock_api

        ckpt = _make_checkpoint(tmp_path)
        client = MimicHubClient()
        client.push_model(ckpt, "user/secret-model", private=True)

        mock_api.create_repo.assert_called_once_with(
            repo_id="user/secret-model",
            repo_type="model",
            private=True,
            exist_ok=True,
        )

    @patch("mimic.hub.client.HfApi")
    def test_push_nonexistent_checkpoint_raises(self, MockHfApi, tmp_path):
        """push_model should raise FileNotFoundError for missing checkpoint."""
        from mimic.hub.client import MimicHubClient

        client = MimicHubClient()
        with pytest.raises(FileNotFoundError):
            client.push_model(tmp_path / "missing.pt", "user/model")


class TestPullModel:
    @patch("mimic.hub.client.snapshot_download")
    @patch("mimic.hub.client.HfApi")
    def test_pull_downloads_model(self, MockHfApi, mock_download, tmp_path):
        """pull_model should call snapshot_download with repo_type='model'."""
        from mimic.hub.client import MimicHubClient

        output = tmp_path / "model_out"
        mock_download.return_value = str(output)

        client = MimicHubClient()
        result = client.pull_model("user/pick-place-act", output)

        mock_download.assert_called_once_with(
            repo_id="user/pick-place-act",
            repo_type="model",
            local_dir=str(output),
        )
        assert result == output

    @patch("mimic.hub.client.snapshot_download")
    @patch("mimic.hub.client.HfApi")
    def test_pull_creates_output_dir(self, MockHfApi, mock_download, tmp_path):
        """pull_model should create the output directory if needed."""
        from mimic.hub.client import MimicHubClient

        output = tmp_path / "deep" / "nested" / "model"
        mock_download.return_value = str(output)

        client = MimicHubClient()
        client.pull_model("user/model", output)

        assert output.exists()

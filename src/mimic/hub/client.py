"""HuggingFace Hub integration for pushing/pulling datasets and models."""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

logger = logging.getLogger(__name__)

DATASET_CARD_TEMPLATE = """\
---
tags:
- robotics
- imitation-learning
- mimic
- lerobot
task_categories:
- robotics
---

# {repo_id}

Robot demonstration dataset uploaded with [Mimic](https://github.com/mimic-robotics/mimic).

## Format

This dataset uses the LeRobot v3 format:

```
meta/
  info.json       # Dataset metadata
  episodes.jsonl  # Episode index
  stats.json      # Normalization statistics
data/
  chunk-NNN/
    episode_NNNNNN.parquet   # Action/state trajectories
videos/
  chunk-NNN/
    <camera>_episode_NNNNNN.mp4  # Camera recordings
```

## Usage

```python
from mimic.hub.client import MimicHubClient

client = MimicHubClient()
client.pull_dataset("{repo_id}", "./my_demos")
```
"""

MODEL_CARD_TEMPLATE = """\
---
tags:
- robotics
- imitation-learning
- mimic
library_name: mimic
pipeline_tag: robotics
---

# {repo_id}

Robot policy checkpoint uploaded with [Mimic](https://github.com/mimic-robotics/mimic).

## Usage

```python
from mimic.hub.client import MimicHubClient

client = MimicHubClient()
client.pull_model("{repo_id}", "./model_dir")
```
"""

EXPECTED_DATASET_DIRS = {"meta", "data", "videos"}


class MimicHubClient:
    """Client for pushing and pulling Mimic datasets and models to/from HuggingFace Hub."""

    def __init__(self, token: str | None = None) -> None:
        """Initialise the client.

        Parameters
        ----------
        token:
            HuggingFace API token.  When *None* the token is read from the
            environment / cached login (``huggingface-cli login``).
        """
        self.api = HfApi(token=token)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def push_dataset(
        self,
        local_path: str | Path,
        repo_id: str,
        *,
        private: bool = False,
    ) -> str:
        """Upload a local dataset directory to HuggingFace Hub.

        Parameters
        ----------
        local_path:
            Path to the dataset root (must contain ``meta/``, ``data/``).
        repo_id:
            Repository identifier, e.g. ``"username/pick-place-demos"``.
        private:
            Whether the repository should be private.

        Returns
        -------
        str
            URL of the created/updated repository.
        """
        local_path = Path(local_path)
        if not local_path.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {local_path}")

        # Validate minimal dataset structure
        meta_dir = local_path / "meta"
        if not meta_dir.is_dir():
            raise ValueError(
                f"Invalid dataset: missing meta/ directory in {local_path}"
            )

        # Create or update the repo
        self.api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )

        # Write dataset card
        card_content = DATASET_CARD_TEMPLATE.format(repo_id=repo_id)
        card_path = local_path / "README.md"
        card_existed = card_path.exists()
        try:
            card_path.write_text(card_content, encoding="utf-8")

            # Upload the entire directory
            self.api.upload_folder(
                folder_path=str(local_path),
                repo_id=repo_id,
                repo_type="dataset",
            )
        finally:
            # Clean up the README if we created it
            if not card_existed and card_path.exists():
                card_path.unlink()

        url = f"https://huggingface.co/datasets/{repo_id}"
        logger.info("Pushed dataset to %s", url)
        return url

    def pull_dataset(
        self,
        repo_id: str,
        output_path: str | Path,
    ) -> Path:
        """Download a dataset from HuggingFace Hub.

        Parameters
        ----------
        repo_id:
            Repository identifier, e.g. ``"username/pick-place-demos"``.
        output_path:
            Local directory to save the dataset into.

        Returns
        -------
        Path
            The path where the dataset was downloaded.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        downloaded = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(output_path),
        )
        logger.info("Pulled dataset to %s", downloaded)
        return Path(downloaded)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    def push_model(
        self,
        checkpoint_path: str | Path,
        repo_id: str,
        *,
        private: bool = False,
    ) -> str:
        """Upload a model checkpoint (and optional config) to HuggingFace Hub.

        The checkpoint ``.pt`` file is uploaded together with any sibling
        ``config.json`` found in the same directory.  A model card is
        generated automatically.

        Parameters
        ----------
        checkpoint_path:
            Path to the ``.pt`` checkpoint file.
        repo_id:
            Repository identifier, e.g. ``"username/pick-place-act"``.
        private:
            Whether the repository should be private.

        Returns
        -------
        str
            URL of the created/updated repository.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        # Create or update the repo
        self.api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )

        # Upload checkpoint
        self.api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=checkpoint_path.name,
            repo_id=repo_id,
            repo_type="model",
        )

        # Upload config.json if present alongside checkpoint
        config_path = checkpoint_path.parent / "config.json"
        if config_path.is_file():
            self.api.upload_file(
                path_or_fileobj=str(config_path),
                path_in_repo="config.json",
                repo_id=repo_id,
                repo_type="model",
            )

        # Upload model card
        card_content = MODEL_CARD_TEMPLATE.format(repo_id=repo_id)
        self.api.upload_file(
            path_or_fileobj=card_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )

        url = f"https://huggingface.co/models/{repo_id}"
        logger.info("Pushed model to %s", url)
        return url

    def pull_model(
        self,
        repo_id: str,
        output_path: str | Path,
    ) -> Path:
        """Download a model from HuggingFace Hub.

        Parameters
        ----------
        repo_id:
            Repository identifier, e.g. ``"username/pick-place-act"``.
        output_path:
            Local directory to save the model files into.

        Returns
        -------
        Path
            The path where the model was downloaded.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        downloaded = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(output_path),
        )
        logger.info("Pulled model to %s", downloaded)
        return Path(downloaded)

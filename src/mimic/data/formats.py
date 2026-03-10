"""Dataset export formats for Mimic.

Supports exporting MimicDataset to:
- LeRobot v3 (native format, copy with validation)
- HDF5 (robomimic compatible)
- RLDS-like (JSON + numpy arrays)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def export_to_lerobot(dataset_path: Path, output_path: Path):
    """Export to LeRobot v3 format (already native -- just copy/validate)."""
    import shutil

    output_path.mkdir(parents=True, exist_ok=True)
    # Our format IS LeRobot v3. Just copy with validation.
    for subdir in ["meta", "data", "videos"]:
        src = dataset_path / subdir
        dst = output_path / subdir
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)


def export_to_hdf5(dataset_path: Path, output_path: Path):
    """Export to HDF5 format (robomimic compatible)."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for HDF5 export: pip install h5py")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = dataset_path / "meta" / "info.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    with h5py.File(str(output_path), "w") as hf:
        hf.attrs["env_name"] = metadata.get("env_name", "unknown")
        hf.attrs["fps"] = metadata.get("fps", 20)

        data_dir = dataset_path / "data" / "chunk-000"
        if not data_dir.exists():
            return

        for parquet_file in sorted(data_dir.glob("episode_*.parquet")):
            table = pq.read_table(parquet_file)
            ep_name = parquet_file.stem  # e.g. episode_000000
            ep_group = hf.create_group(f"data/{ep_name}")

            # Convert columns
            for col_name in ["state", "action", "joint_pos", "joint_vel"]:
                if col_name in table.column_names:
                    col = table.column(col_name)
                    arr = np.array([row.as_py() for row in col])
                    ep_group.create_dataset(col_name, data=arr)

            if "reward" in table.column_names:
                ep_group.create_dataset("reward", data=table.column("reward").to_numpy())
            if "done" in table.column_names:
                ep_group.create_dataset("done", data=table.column("done").to_numpy())


def export_to_rlds(dataset_path: Path, output_path: Path):
    """Export to RLDS-like format (JSON + numpy arrays)."""
    output_path.mkdir(parents=True, exist_ok=True)
    meta_path = dataset_path / "meta" / "info.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    data_dir = dataset_path / "data" / "chunk-000"
    if not data_dir.exists():
        return

    episodes = []
    for parquet_file in sorted(data_dir.glob("episode_*.parquet")):
        table = pq.read_table(parquet_file)
        episode = {"steps": []}
        for i in range(table.num_rows):
            step = {}
            for col_name in table.column_names:
                val = table.column(col_name)[i].as_py()
                step[col_name] = val
            episode["steps"].append(step)
        episodes.append(episode)

    # Save as JSON (RLDS-like structure)
    with open(output_path / "episodes.json", "w") as f:
        json.dump({"metadata": metadata, "episodes": episodes}, f)

    # Also save numpy arrays
    for i, ep in enumerate(episodes):
        ep_dir = output_path / f"episode_{i:06d}"
        ep_dir.mkdir(exist_ok=True)
        steps = ep["steps"]
        for key in ["state", "action"]:
            if key in steps[0]:
                arr = np.array([s[key] for s in steps])
                np.save(ep_dir / f"{key}.npy", arr)

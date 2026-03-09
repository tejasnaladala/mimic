"""Video encoding/decoding utilities for Mimic datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def encode_video(
    frames: list[np.ndarray],
    output_path: str | Path,
    fps: int = 20,
    codec: str = "h264",
):
    """Encode a list of RGB frames to an MP4 video."""
    import av

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        return

    h, w = frames[0].shape[:2]
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream(codec, rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"

    for frame_data in frames:
        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


def decode_video(video_path: str | Path) -> list[np.ndarray]:
    """Decode an MP4 video to a list of RGB frames."""
    import av

    frames = []
    container = av.open(str(video_path))
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return frames

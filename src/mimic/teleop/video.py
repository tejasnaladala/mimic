from __future__ import annotations

import asyncio

from aiortc import VideoStreamTrack
from av import VideoFrame


class MuJoCoVideoTrack(VideoStreamTrack):
    """Streams MuJoCo renders as WebRTC video frames."""

    kind = "video"

    def __init__(self, frame_queue: asyncio.Queue):
        super().__init__()
        self._queue = frame_queue

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        img = await self._queue.get()
        frame = VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame

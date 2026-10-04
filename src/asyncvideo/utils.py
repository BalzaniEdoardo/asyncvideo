from __future__ import annotations

from concurrent.futures import Future
from enum import IntEnum, StrEnum
from multiprocessing.shared_memory import SharedMemory
from typing import TypeAlias

import av
import numpy as np
from numpy.typing import NDArray

UInt8Array: TypeAlias = NDArray[np.uint8]
TupleYUV: TypeAlias = tuple[UInt8Array, UInt8Array, UInt8Array]
FutureArray: TypeAlias = Future[UInt8Array | TupleYUV]
SharedMemRGB: TypeAlias = tuple[SharedMemory]
SharedMemYUV: TypeAlias = tuple[SharedMemory, SharedMemory, SharedMemory]


class Colorspace(StrEnum):
    rgb24 = "rgb24"
    yuv420p = "yuv420p"


class ReaderError(IntEnum):
    """Outcome of a single reader request, as sent back to the parent process.

    Only this category crosses the process boundary; the full traceback is
    logged in the worker. Keeping the response a pair of ints means it is
    always serializable, so the reply can never be lost to a pickling failure
    in ``multiprocessing.Queue``'s feeder thread — which would leave the caller
    waiting on a future that never resolves.
    """

    ok = 0
    unknown = 1
    type_error = 2
    value_error = 3
    index_error = 4
    key_error = 5
    memory_error = 6
    os_error = 7


def create_shared_memory(
    frame: av.VideoFrame, n_frames: int = 1, yuv_packed: bool = False
) -> SharedMemYUV | SharedMemRGB:
    colorspace = frame.format.name
    # yuvj420p (full-range JPEG) is storage-identical to yuv420p
    if colorspace == "yuvj420p":
        colorspace = "yuv420p"

    rows, cols = frame.height, frame.width

    if colorspace == "yuv420p":
        if yuv_packed:
            # packed shape
            rows = rows * 3 // 2
            return (SharedMemory(create=True, size=rows * cols * 3 * n_frames),)

        y = SharedMemory(create=True, size=rows * cols * n_frames)

        rows_chroma, cols_chroma = (
            frame.format.chroma_height(),
            frame.format.chroma_width(),
        )
        u = SharedMemory(create=True, size=rows_chroma * cols_chroma * n_frames)
        v = SharedMemory(create=True, size=rows_chroma * cols_chroma * n_frames)

        return y, u, v

    elif colorspace == "rgb24":
        return (SharedMemory(create=True, size=rows * cols * 3 * n_frames),)

    else:
        raise ValueError(
            f"only rgb24 and yuv420p colorspaces are currently supported, "
            f"provided video with colorspace: {colorspace}"
        )


def create_buffers(
    shared_mems: SharedMemYUV | SharedMemRGB,
    colorspace: Colorspace,
    shape_frame: tuple[int, int],
    shape_chroma: tuple[int, int] | None,
    n_frames: int = 1,
    yuv_packed: bool = False,
) -> UInt8Array | TupleYUV:
    if colorspace == Colorspace.yuv420p:
        if yuv_packed:
            buffer = np.ndarray(
                shape=(n_frames, shape_frame[0] * 3 // 2, shape_frame[1]),
                dtype=np.uint8,
                buffer=shared_mems[0].buf,
            )
            return buffer

        buffer_y = np.ndarray(
            shape=(n_frames, *shape_frame), dtype=np.uint8, buffer=shared_mems[0].buf
        )
        buffer_u = np.ndarray(
            shape=(n_frames, *shape_chroma), dtype=np.uint8, buffer=shared_mems[1].buf
        )
        buffer_v = np.ndarray(
            shape=(n_frames, *shape_chroma), dtype=np.uint8, buffer=shared_mems[2].buf
        )

        return buffer_y, buffer_u, buffer_v

    elif colorspace == Colorspace.rgb24:
        # rgb
        buffer_rgb = np.ndarray(
            shape=(n_frames, *shape_frame), dtype=np.uint8, buffer=shared_mems[0].buf
        )

        return buffer_rgb

    else:
        raise ValueError(
            f"only rgb24 and yuv420p colorspaces are currently supported, "
            f"provided video with colorspace: {colorspace}"
        )

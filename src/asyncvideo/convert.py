"""Colorspace conversion helpers.

Conversion is delegated to PyAV rather than implemented with a hand-written
matrix: the correct YUV to RGB coefficients depend on the stream's colorspace
and colour range, and libav already implements that. Every path here therefore
funnels through `av.VideoFrame`, which is also the format the readers hand back
when ``pixel_format`` is ``None``.
"""

from __future__ import annotations

from typing import Literal

import av
import numpy as np
from numpy.typing import NDArray

__all__ = ["to_rgb"]

PixelFormat = Literal["rgb24", "yuv420p", "yuvj420p", "yuv444p"]

RGB24: PixelFormat = "rgb24"
YUV420P: PixelFormat = "yuv420p"
YUVJ420P: PixelFormat = "yuvj420p"
YUV444P: PixelFormat = "yuv444p"

# per-format number of dimensions of a *single* frame, used to tell a single
# frame from a stack of frames
_SINGLE_NDIM: dict[PixelFormat, int] = {RGB24: 3, YUV420P: 2, YUVJ420P: 2, YUV444P: 3}


def _is_rgb(arr: NDArray) -> bool:
    return arr.ndim in (3, 4) and arr.shape[-1] == 3


def _infer_pixel_format(arr: NDArray) -> PixelFormat:
    """Guess the layout of ``arr``.

    ``yuv444p`` is deliberately not inferred: a single ``yuv444p`` frame is
    ``(3, H, W)``, which is indistinguishable from a stack of packed
    ``yuv420p`` frames. Callers must name that one explicitly.
    """
    if _is_rgb(arr):
        return RGB24
    if arr.ndim in (2, 3):
        return YUV420P
    raise ValueError(
        f"cannot infer the pixel format of an array with shape {arr.shape}; "
        f"pass pixel_format explicitly"
    )


def _planes_to_packed(y: NDArray, u: NDArray, v: NDArray) -> NDArray:
    """Pack ``(Y, U, V)`` planes into the flat ``yuv420p`` layout PyAV expects.

    ``VideoFrame.to_ndarray()`` on a ``yuv420p`` frame returns the frame buffer
    reshaped to ``(H * 3 // 2, W)`` — Y first, then U, then V. Concatenating the
    flattened planes in that order reproduces it without hardcoding how many
    rows the chroma planes occupy.
    """
    if y.ndim == 2:
        rows, cols = y.shape
        return np.concatenate([y.ravel(), u.ravel(), v.ravel()]).reshape(
            rows * 3 // 2, cols
        )
    if y.ndim == 3:
        n, rows, cols = y.shape
        packed = np.empty((n, rows * 3 // 2, cols), dtype=y.dtype)
        for i in range(n):
            packed[i] = np.concatenate(
                [y[i].ravel(), u[i].ravel(), v[i].ravel()]
            ).reshape(rows * 3 // 2, cols)
        return packed
    raise ValueError(
        f"expected planes with 2 or 3 dimensions, got {y.ndim} (shape {y.shape})"
    )


def _single_to_rgb(arr: NDArray, pixel_format: PixelFormat) -> NDArray:
    # from_ndarray needs a contiguous uint8 buffer; the plane helpers hand back
    # strided views onto padded rows.
    frame = av.VideoFrame.from_ndarray(
        np.ascontiguousarray(arr, dtype=np.uint8), format=pixel_format
    )
    return frame.to_ndarray(format=RGB24)


def to_rgb(frames, from_format: PixelFormat | None = None) -> NDArray[np.uint8]:
    """
    Convert any of the readers' outputs to RGB.

    See Also
    --------
    VideoHandler.to_rgb, AsyncVideoReader.to_rgb
        Method forms that supply ``from_format`` from the reader's own
        configuration, so it never has to be passed by hand.

    Parameters
    ----------
    frames :
        One of:

        - an `av.VideoFrame`, or a list/tuple of them (what the readers return
          when ``pixel_format`` is ``None``);
        - a ``(Y, U, V)`` tuple of plane arrays, as `AsyncVideoReader` returns
          for a ``yuv420p`` video. Planes may be ``(H, W)`` or ``(n, H, W)``;
        - a packed ``yuv420p`` array, ``(H * 3 // 2, W)`` or
          ``(n, H * 3 // 2, W)``;
        - a ``yuv444p`` array, ``(3, H, W)`` or ``(n, 3, H, W)``;
        - an array that is already RGB, ``(H, W, 3)`` or ``(n, H, W, 3)``,
          which is returned unchanged.
    from_format :
        Layout ``frames`` is being converted *from*, when it is an array.
        Inferred when omitted, except for ``"yuv444p"``: a single ``yuv444p``
        frame is ``(3, H, W)``, which cannot be told apart from a stack of
        packed ``yuv420p`` frames, so that case must be named explicitly.
        Ignored when ``frames`` is an `av.VideoFrame` or a ``(Y, U, V)``
        triple, which describe their own layout.

    Returns
    -------
    :
        ``(H, W, 3)`` uint8 for a single frame, ``(n, H, W, 3)`` for a stack.

    Examples
    --------
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> from asyncvideo import AsyncVideoReader, to_rgb  # doctest: +SKIP
    >>> reader = AsyncVideoReader("example.mp4")  # doctest: +SKIP
    >>> planes = reader[(slice(0, 1),)].result()  # doctest: +SKIP
    >>> plt.imshow(to_rgb(planes)[0])  # doctest: +SKIP
    """
    if isinstance(frames, av.VideoFrame):
        return frames.to_ndarray(format=RGB24)

    if isinstance(frames, (list, tuple)):
        if len(frames) == 0:
            raise ValueError("cannot convert an empty sequence of frames")

        if all(isinstance(f, av.VideoFrame) for f in frames):
            return np.stack([f.to_ndarray(format=RGB24) for f in frames])

        # otherwise the only sequence we accept is a (Y, U, V) plane triple
        if len(frames) != 3:
            raise TypeError(
                f"expected an av.VideoFrame, a sequence of them, or a (Y, U, V) "
                f"plane triple; got a sequence of length {len(frames)}"
            )
        y, u, v = (np.asarray(plane) for plane in frames)
        return to_rgb(_planes_to_packed(y, u, v), from_format=YUV420P)

    arr = np.asarray(frames)

    if from_format is None:
        fmt: PixelFormat = _infer_pixel_format(arr)
    elif from_format not in _SINGLE_NDIM:
        raise ValueError(
            f"unsupported from_format {from_format!r}; "
            f"expected one of {sorted(_SINGLE_NDIM)}"
        )
    else:
        fmt = from_format

    if fmt == RGB24:
        return arr

    single_ndim = _SINGLE_NDIM[fmt]
    if arr.ndim == single_ndim:
        return _single_to_rgb(arr, fmt)
    if arr.ndim == single_ndim + 1:
        return np.stack([_single_to_rgb(a, fmt) for a in arr])
    raise ValueError(
        f"array with shape {arr.shape} is not a single {fmt} frame "
        f"({single_ndim} dimensions) or a stack of them ({single_ndim + 1})"
    )

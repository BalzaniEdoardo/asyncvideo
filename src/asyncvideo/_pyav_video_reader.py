"""
Base class for audio and video handling.

Handles opening/closing the stream, seeking, and keyframe extraction.
"""

from __future__ import annotations

import abc
import logging
import pathlib
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from typing import Literal

import av
import numpy as np
from numpy.typing import NDArray

from .convert import to_rgb
from .exceptions import _Superseded

logger = logging.getLogger(__name__)

# Number of packets to buffer before flushing to the index for codecs without
# B-frames (where packet PTS are already in display order).
_INDEX_FLUSH_EVERY = 64

# How many progressively earlier restarts to try when a seek lands past the frame
# that was asked for. Each one rewinds a further keyframe, so a small number
# covers the realistic cases without turning one bad seek into a scan of the
# whole file.
_MAX_REWINDS = 3


class FrameBuffer:
    """Fixed-size FIFO cache mapping frame index → raw av.VideoFrame.

    Frames are stored in their native pixel format; conversion happens on
    retrieval, matching the existing behaviour of VideoHandler.

    Parameters
    ----------
    maxsize :
        Maximum number of frames to keep. When full the oldest-inserted
        entry is evicted before adding a new one.
    """

    def __init__(self, maxsize: int = 30) -> None:
        self._maxsize = maxsize
        self._cache: dict[int, av.VideoFrame] = {}
        self._order: deque[int] = deque()

    def get(self, idx: int) -> av.VideoFrame | None:
        """Return the cached frame for *idx*, or ``None`` on a miss."""
        return self._cache.get(idx)

    def put(self, idx: int, frame: av.VideoFrame) -> None:
        """Insert *frame* under *idx*, evicting the oldest entry if full."""
        if idx in self._cache:
            return
        if len(self._cache) == self._maxsize:
            evict = self._order.popleft()
            del self._cache[evict]
        self._cache[idx] = frame
        self._order.append(idx)

    def __contains__(self, idx: int) -> bool:
        return idx in self._cache

    def __repr__(self):
        if len(self._cache) <= 1:
            return "".join(
                ["FrameBuffer("] + [f"{k}: {v}" for k, v in self._cache.items()] + [")"]
            )
        return "".join(
            ["FrameBuffer(\n"]
            + [f"\t{k}: {v}\n" for k, v in self._cache.items()]
            + [")"]
        )


def _needs_flush(
    count_keyframes: int, temp: list, has_b_frames: bool, n_b_frames: int = 1
) -> bool:
    """True when the buffered GOP / batch is ready to commit to the index.

    Parameters
    ----------
    count_keyframes :
        Number of keyframe in a frame block.
    temp :
        A list of extracted pts.
    has_b_frames:
        True if the codec has B-frames.
    n_b_frames:
        Number of B-frames.
    """
    if has_b_frames:
        return (count_keyframes == n_b_frames) and bool(temp)
    return len(temp) >= _INDEX_FLUSH_EVERY


def pyav_trim_plane(plane, bytes_per_pixel=1, dtype="uint8"):
    """
    Adapted from pyav

    Return the useful part of the VideoPlane as a strided array.

    We are simply creating a view that discards any padding which was added for
    alignment.
    """

    dtype_obj = np.dtype(dtype)
    total_line_size = abs(plane.line_size)
    itemsize = dtype_obj.itemsize
    channels = bytes_per_pixel // itemsize

    if channels == 1:
        shape = (plane.height, plane.width)
        strides = (total_line_size, itemsize)
    else:
        shape = (plane.height, plane.width, channels)
        strides = (total_line_size, bytes_per_pixel, itemsize)

    return np.ndarray(shape, dtype=dtype_obj, buffer=plane, strides=strides)


class BaseAudioVideo:
    def __init__(
        self,
        path: str | pathlib.Path,
    ) -> None:
        self.file_path = pathlib.Path(path)
        self.container = av.open(path)
        self._running = True

        # initialize index for last decoded frame
        # if sampling of other signals (LFP) is much denser, multiple times the frame
        # is unchanged, so cache the idx
        self.last_loaded_idx = None

        self._lock = threading.Lock()

        self._keyframe_pts = []
        self._pts_keyframe_ready = threading.Event()
        self._keyframe_thread = threading.Thread(
            target=self._extract_keyframes_pts, daemon=True
        )
        self._keyframe_thread.start()

    @abc.abstractmethod
    def _ts_to_pts(self, ts: float) -> int:
        pass

    @abc.abstractmethod
    def _extract_keyframes_pts(self):
        pass

    def _need_seek_call(self, current_frame_pts, target_frame_pts):
        if current_frame_pts is None:
            return True

        with self._lock:
            if len(self._keyframe_pts) == 0:
                return True
            # While the keyframe thread is still running we may not yet know
            # about a keyframe that sits between current position and target.
            # Seek conservatively so we don't miss it.
            # Once the thread is done the list is complete: no keyframe beyond
            # the last known one exists, so the absence of one is not a reason
            # to seek — we can stream forward safely.
            if (
                not self._pts_keyframe_ready.is_set()
                and self._keyframe_pts[-1] < target_frame_pts
            ):
                return True

        # roll back the stream if audiovideo is scrolled backwards
        if current_frame_pts > target_frame_pts:
            return True

        # find the closest keyframe pts before a given frame
        idx = np.searchsorted(self._keyframe_pts, target_frame_pts, side="right")
        closest_keyframe_pts = self._keyframe_pts[max(0, idx - 1)]

        # seek forward only if there is a keyframe between current position
        # and the target (i.e. a closer starting point exists).
        return closest_keyframe_pts > current_frame_pts

    def close(self):
        """Close the audio-video stream."""
        self._running = False
        threads = ["_index_thread", "_keyframe_thread"]
        for thread_name in threads:
            # index thread is only for video frames
            thread = getattr(self, thread_name, None)
            if thread is not None and thread.is_alive():
                thread.join(timeout=1)
        try:
            self.container.close()
        except Exception:
            logger.exception("Failed to close the audiovideo stream.")
        finally:
            # dropping refs to fully close av.InputContainer
            self.container = None
            self.stream = None

    # context protocol
    # (with AudioHandler(path) as audiovideo ensure closing)
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class VideoHandler(BaseAudioVideo):
    """
    Random-access video reader with timestamp-aware seeking.

    This class wraps a PyAV container to provide precise, timestamp-based
    random access to frames. It can return either `av.VideoFrame` objects
    or RGB float arrays in the shape ``(H, W, 3)`` normalized to ``[0, 1]``.
    Internally, a background thread builds an index of presentation timestamps
    (PTS) for fast seeks, with a fallback to on-the-fly indexing when the total
    frame count is unknown.

    Parameters
    ----------
    video_path :
        Path to the video file.
    stream_index :
        Index of the video stream to use, default is 0.
    time :
        Experimental timestamps for each frame (seconds). If ``None``, a
        uniform grid is generated from the stream's average rate.
    pixel_format :
        PyAV pixel format string for decoded frames. Supported values are
        ``"rgb24"``, ``"yuv420p"`` and ``"yuv444p"``. The default is ``None``,
        which skips conversion and returns raw `av.VideoFrame` instances; pass
        an explicit format to get numpy arrays instead.
    buffer_size :
        Number of recently decoded frames to keep in the FIFO frame buffer.
        On a cache hit the frame is returned without any seeking or decoding.
        Default is 30 (roughly 1 s at 30 fps).
    abort_decoding :
        Called once per decoded frame while scanning towards a target. Returning
        ``True`` stops the scan and raises ``_Superseded``; the frames decoded so
        far stay cached and the read position is kept, so a later request can
        carry on from there. ``None`` (the default) never aborts.

    Examples
    --------
    >>> from asyncvideo import VideoHandler
    >>> # Pass a pixel_format to get numpy arrays; the default returns
    >>> # raw av.VideoFrame objects instead.
    >>> vh = VideoHandler("example.mp4", pixel_format="rgb24")  # doctest: +SKIP
    >>> # Shape: (n_frames, height, width, channels)
    >>> vh.shape  # doctest: +SKIP
    (100, 480, 640, 3)
    >>> # Get the frame at 1.5 seconds.
    >>> frame = vh.get(1.5)  # doctest: +SKIP
    >>> # Shape: (height, width, channels)
    >>> frame.shape  # doctest: +SKIP
    (480, 640, 3)
    >>> # Get frames from the second to the 10th, every other frame.
    >>> frame_sequence = vh[1:10:2]  # doctest: +SKIP
    >>> # Shape: (n_samples, height, width, channels)
    >>> frame_sequence.shape  # doctest: +SKIP
    (5, 480, 640, 3)
    """

    def __init__(
        self,
        video_path: str | pathlib.Path,
        stream_index: int = 0,
        time: NDArray | None = None,
        pixel_format: Literal["rgb24", "yuv420p", "yuv444p"] | None = None,
        abort_decoding: Callable[[], bool] | None = None,
        buffer_size: int = 30,
    ) -> None:
        super().__init__(video_path)
        self._buffer = FrameBuffer(maxsize=buffer_size)
        # pts of the last frame *actually decoded* from the stream — used for
        # seek decisions.  current_frame can be updated by buffer / cache hits
        # without advancing the stream, so it must not be used for this purpose.
        self._stream_pts: int | None = None
        # True once the stream has been decoded to exhaustion: the decoder is
        # flushed and must be re-seeked before it will accept another packet
        self._at_eof = False
        # Frame-level iterator over the stream, kept alive between reads and
        # dropped only by ``_seek``. It has to outlive a single read: a packet
        # can decode to more than one frame (AV1 reorders with
        # show_existing_frame OBUs rather than DTS != PTS, so libdav1d emits two
        # frames from one packet), and a generator abandoned after the first of
        # those frames takes the rest with it.
        self._decoder = None
        self.stream = self.container.streams.video[stream_index]
        self.stream_index = stream_index
        self.pixel_format = pixel_format

        # Frame times are resolved once, by the index thread, and published through
        # this future. Deriving them needs every frame's PTS, which is only known
        # when indexing completes -- so rather than hand out a provisional guess
        # from the nominal frame rate and silently change it later, ``time``
        # blocks on the future and returns a single, real answer. Callers that
        # never ask for a timestamp never wait for it.
        self._time_provided = time is not None
        self._time_input = None if time is None else np.asarray(time)
        self._time_future: Future[NDArray] = Future()

        # Most containers declare their frame count, so a mismatched ``time``
        # array can be rejected here — at the line that passed it — rather than
        # later, from whichever call first needs a timestamp. Containers that
        # declare nothing (vp9/webm report 0) are still checked by
        # ``_resolve_time`` once the indexer has counted the frames, which stays
        # the authoritative check: a header count can disagree with reality.
        if self._time_provided and self.stream.frames > 0:
            self._check_time_length(len(self._time_input), self.stream.frames)

        # initialize index for last decoded frame
        # if sampling of other signals (LFP) is much denser, multiple times the frame
        # is unchanged, so cache the idx
        self.last_loaded_idx = None

        # initialize current frame
        self.current_frame: av.VideoFrame | None = None

        # measured lazily from the first decoded frame when pixel_format is None,
        # since the native layout is only knowable from a real frame
        self._native_frame_shape: tuple[int, ...] | None = None

        if self.file_path.suffix == ".mkv":
            # mkv time is rounded to 3 digits, at least in the example video
            # generated by tests/generate_numbered_video.py
            self.round_fn = lambda x: np.round(x, 3)
        else:
            self.round_fn = lambda x: x

        # These will be initialized in the thread once n_frames is known
        self.all_pts: np.ndarray | list = []
        self.all_times = None
        self.key_mask = None

        self._i = 0  # number of committed (valid) PTS entries
        # None means the total frame count is not yet known (e.g. vp9/webm);
        # set to the final count by _build_index before signalling _index_ready.
        self._n_frames: int | None = (
            self.stream.frames if self.stream.frames > 0 else None
        )
        self._index_thread = threading.Thread(target=self._build_index, daemon=True)

        self._index_ready = threading.Event()
        self._index_thread.start()
        self._abort_decoding = abort_decoding
        # decode first frame
        self.__getitem__(0)

    @staticmethod
    def _ts_to_index(ts: float, time: NDArray) -> int:
        """
        Return the index of the frame whose experimental time is just before (or equal to) `ts`.

        Parameters
        ----------
        ts : float
            Experimental timestamp to match.
        time : NDArray
            Array of experimental timestamps, assumed sorted in ascending order,
            with one entry per frame.

        Returns
        -------
        idx : int
            Index of the frame with time <= `ts`. Clipped to [0, len(time) - 1].

        Notes
        -----
        - If `ts` is smaller than all values in `time`, returns 0.
        - If `ts` is greater than all values in `time`, returns `len(time) - 1`.
        """
        idx = np.searchsorted(time, ts, side="right") - 1
        return np.clip(idx, 0, len(time) - 1)

    def _extract_keyframe_times_and_points(
        self, video_path: str | pathlib.Path, stream_index: int = 0, first_only=False
    ) -> tuple[NDArray, NDArray] | None:
        """
        Extract the indices and timestamps of keyframes from a video file.

        This function decodes the video while skipping non-keyframes, and records:
        - The index of each keyframe in the full video frame sequence
        - The "Presentation Time Stamp" to each keyframe.

        It is typically intended to run in a background thread during
        initialization of a ``VideoHandler``, and supports optimized seeking:

        - When the requested frame (based on experimental time) is before the
          current playback position, seeking backward is necessary.

        - When the requested frame is beyond the next known keyframe, seeking
          forward to the closest keyframe is more efficient than decoding all
          intermediate frames.

        Parameters
        ----------
        video_path : str or pathlib.Path
            The path to the video file.
        stream_index:
            The index of the video stream.
        first_only:
            If true, return the first keyframe only. Used at initialization.

        Returns
        -------
        keyframe_points : NDArray[float]
            The point number of the frame.

        keyframe_timestamps : NDArray[float]
            The timestamp of the frame.
        """
        keyframe_timestamp = []
        keyframe_pts = []

        with av.open(video_path) as container:
            stream = container.streams.video[stream_index]
            stream.codec_context.skip_frame = "NONKEY"

            for frame in container.decode(stream):
                if not self._running:
                    return
                keyframe_timestamp.append(frame.time)
                keyframe_pts.append(frame.pts)
                if first_only:
                    break

        return np.asarray(keyframe_pts), np.asarray(keyframe_timestamp, dtype=float)

    def _extract_keyframes_pts(self):
        try:
            with av.open(self.file_path) as container:
                stream = container.streams.video[0]
                for packet in container.demux(stream):
                    if not self._running:
                        return
                    if packet.is_keyframe:
                        with self._lock:
                            self._keyframe_pts.append(packet.pts)
        except Exception:
            logger.exception("Keyframe thread error")
        finally:
            self._pts_keyframe_ready.set()

    def _build_index(self):
        try:
            with av.open(self.file_path) as container:
                stream = container.streams.video[self.stream_index]
                n_frames = stream.frames
                ctx = stream.codec_context
                has_b_frames = bool(ctx.has_b_frames)
                # guard against max_b_frames set to None for non-b-frame codecs
                max_b_frames = max(getattr(ctx, "max_b_frames", 1) or 1, 1)
                process = sorted if has_b_frames else lambda x: x
                temp = []
                # setup config for fixed-size and variable size index.
                if n_frames > 0:
                    # preallocate indices
                    with self._lock:
                        self.all_pts = np.empty(n_frames, dtype=np.int64)

                    def update(extracted_pts):
                        chunk = process(extracted_pts)
                        with self._lock:
                            self.all_pts[self._i : self._i + len(chunk)] = chunk
                            self._i += len(chunk)
                        extracted_pts.clear()
                else:

                    def update(extracted_pts):
                        chunk = process(extracted_pts)
                        with self._lock:
                            self.all_pts.extend(chunk)
                            self._i = len(self.all_pts)
                        extracted_pts.clear()

                # extraction loop: do not decode but sort and trim if needed.
                count_key_frames = 0
                for packet in container.demux(stream):
                    if not self._running:
                        return
                    if packet.pts is None or packet.pts < 0:
                        continue

                    count_key_frames += packet.is_keyframe
                    if _needs_flush(count_key_frames, temp, has_b_frames, max_b_frames):
                        update(temp)
                        count_key_frames = 0
                    temp.append(packet.pts)

                if temp:
                    update(temp)
                with self._lock:
                    self.all_pts = np.asarray(self.all_pts[: self._i], dtype=np.int64)

        except Exception:
            logger.exception("Index thread error")
        finally:
            self._n_frames = self._i
            self._resolve_time()
            self._index_ready.set()

    @staticmethod
    def _check_time_length(n_times: int, n_frames: int) -> None:
        """Raise if a provided ``time`` array does not have one entry per frame."""
        if n_times != n_frames:
            raise ValueError(
                f"the provided time array has length {n_times}, but the video has "
                f"{n_frames} frames; pass one timestamp per frame"
            )

    def _resolve_time(self):
        """Publish the frame times through ``_time_future``.

        Always resolves the future, on every exit path of the index thread: a
        caller reading ``time`` is blocked on it, so leaving it pending would hang
        them rather than fail them.
        """
        if self._time_future.done():
            return
        try:
            if self._time_provided:
                self._check_time_length(len(self._time_input), self._i)
                self._time_future.set_result(self._time_input)
                return

            # Real frame times from the stream's own presentation timestamps.
            # all_pts is in stream time_base units and sorted into display order
            # by the indexer, so scaling it gives absolute, monotonic seconds --
            # accurate for variable frame rate video, unlike a uniform grid.
            pts = np.asarray(self.all_pts[: self._i], dtype=np.int64)
            if len(pts) != self._i:
                raise ValueError(
                    f"indexing found {self._i} frames but only {len(pts)} "
                    f"presentation timestamps"
                )
            self._time_future.set_result(pts * float(self.stream.time_base))
        except BaseException as exc:  # noqa: BLE001 - must not leave callers hanging
            self._time_future.set_exception(exc)

    def _get_frame_idx(self, pts: int) -> tuple[int, bool]:
        """
        Get the frame index from the presentation time stamp.

        Parameters
        ----------
        pts:
            The presentation time stamp of the frame.

        Returns
        -------
        idx:
            The frame index corresponding to the given pts.
        use_time:
            If true, search using presentation time in seconds, otherwise use pts.

        """
        # Wait until enough index is available
        # Estimate pts from index (using filled index if available)
        with self._lock:
            done = self._i > 0 and self.all_pts[self._i - 1] > pts
        if done:
            # the pts for this timestamp has been filled
            idx = np.searchsorted(self.all_pts[: self._i], pts, side="left")
            use_time = False
        else:
            # keep going until at least two frames have been decoded by the thread
            while True:
                with self._lock:
                    if self._i > 1:
                        break
                time.sleep(0.001)
            # use recent history to get the step estimate
            with self._lock:
                # Linear extrapolation from available pts (use last 10 steps for an estimate)
                start, stop = max(self._i - 10, 0), self._i
                avg_step = np.mean(np.diff(self.all_pts[start:stop]))
                idx = int((pts - self.all_pts[0]) / avg_step)
                use_time = True
        return idx, use_time

    def _get_target_frame_pts(self, idx: int) -> tuple[int, bool]:
        """
        Get the target frame presentation time stamp from frame index.

        Parameters
        ----------
        idx:
            The frame index.

        Returns
        -------
        target_pts:
            The target frame presentation time stamp corresponding to the frame index.
        use_time:
            If true, search using presentation time in seconds, otherwise use pts.

        """
        # Wait until enough index is available
        # Estimate pts from index (using filled index if available)
        with self._lock:
            done = self._i > idx
        if done:
            # the pts for this timestamp has been filled
            target_pts = self.all_pts[idx]
            use_time = False
        else:
            # keep going until at least two frames have been decoded by the thread
            while True:
                with self._lock:
                    if self._i > 1:
                        break
                time.sleep(0.001)
            # use recent history to get the step estimate
            with self._lock:
                # Linear extrapolation from available pts (use last 10 steps for an estimate)
                start, stop = max(self._i - 10, 0), self._i
                avg_step = np.mean(np.diff(self.all_pts[start:stop]))
                target_pts = int(self.all_pts[0] + avg_step * idx)
                use_time = True
        return target_pts, use_time

    def _get_key_frame(self, backward) -> av.VideoFrame | NDArray:
        idx = self.last_loaded_idx
        if idx is None:
            # fallback to safe keyframe
            self._pts_keyframe_ready.wait(2.0)
            if len(self._keyframe_pts) > 0:
                idx = self._get_frame_idx(self._keyframe_pts[0])[0] + 1
            else:
                idx = 0  # safe fallback

        # Get the pts of the last loaded index
        target_pts, _ = self._get_target_frame_pts(idx)

        # Seek the next or previous keyframe based on the direction
        with self._lock:
            delta = max(np.mean(np.diff(self._keyframe_pts[:10])) // 2, 1)
        try:
            self.container.seek(
                int(
                    target_pts + (-delta if backward else delta)
                ),  # if you're on top of a key frame, seek does not move no matter what
                backward=backward,
                any_frame=False,
                stream=self.stream,
            )
        except av.error.PermissionError:
            # seek backward at the end of the file
            self.container.seek(
                int(target_pts),
                backward=True,
                any_frame=False,
                stream=self.stream,
            )
        # This seeks and then demuxes by hand, so the shared iterator no longer
        # describes where the stream is.
        self._at_eof = False
        self._decoder = None

        # Decode the next frame, which should be a keyframe
        frame = next(
            frame
            for packet in self.container.demux(self.stream)
            if packet is not None
            for frame in packet.decode()
        )

        self.current_frame = frame

        # Get the index of the key frame
        self.last_loaded_idx = self._get_frame_idx(frame.pts)[0]

        # Return both
        return (
            self.current_frame.to_ndarray(format=self.pixel_format)
            if self.pixel_format is not None
            else self.current_frame,
            self.last_loaded_idx,
        )

    def get(self, ts: float) -> av.VideoFrame | NDArray:
        """
        Return the frame at (or immediately preceding) a timestamp.

        Parameters
        ----------
        ts : float
            Target time in seconds.

        Returns
        -------
        :
            If ``pixel_format`` is not ``None``, returns a ``np.ndarray``
            with shape matching the format (e.g. ``(H, W, 3)`` for ``"rgb24"``
            or ``(H*3//2, W)`` for ``"yuv420p"``). Otherwise returns an
            `av.VideoFrame`.

        Notes
        -----
        - Seeks to the closest keyframe behind ``ts`` and decodes forward
          until the target is reached.
        - Uses an internal cache: if the requested frame index matches the
          previously decoded one, the cached frame is returned.
        """
        return self._get_by_index(self._ts_to_index(ts, self.time))

    def _get_by_index(self, idx: int):
        """
        Return the frame at frame index ``idx``.

        The shared body of `get` and ``__getitem__``: everything past resolving a
        timestamp to an index is the same for both. Keeping it a separate method is
        what lets ``__getitem__`` reuse it without having to tell `get` to
        reinterpret its argument as an index -- which is state the two calls would
        otherwise have to share, and which could not then be relied on across
        threads or async tasks.
        """
        if idx == self.last_loaded_idx:
            return (
                self.current_frame.to_ndarray(format=self.pixel_format)
                if self.pixel_format is not None
                else self.current_frame
            )

        cached = self._buffer.get(idx)
        if cached is not None:
            self.current_frame = cached
            self.last_loaded_idx = idx
            return (
                cached.to_ndarray(format=self.pixel_format)
                if self.pixel_format is not None
                else cached
            )

        target_pts, use_time = self._get_target_frame_pts(idx)

        if (
            self._stream_pts is None
            # a decoder left at EOF cannot be decoded from, whatever the target
            or self._at_eof
            or self._need_seek_call(self._stream_pts, target_pts)
        ):
            self._seek(target_pts)

        # Decode forward from the keyframe until the frame just before (or equal to) target_pts
        _, preceding_frame = self._decode_and_check_frames(use_time, target_pts, idx)

        if preceding_frame is not None:
            self.last_loaded_idx = idx
            self.current_frame = preceding_frame
            self._stream_pts = preceding_frame.pts
            self._buffer.put(idx, preceding_frame)

        return (
            self.current_frame.to_ndarray(format=self.pixel_format)
            if self.pixel_format is not None
            else self.current_frame
        )

    def _seek(self, target_pts: int) -> None:
        """Seek to the keyframe at or before ``target_pts``.

        Seeking flushes the decoder, so it is also how the reader recovers from
        having previously run the stream to EOF.
        """
        self.container.seek(
            int(target_pts), backward=True, any_frame=False, stream=self.stream
        )
        self._at_eof = False
        # Frames still queued in the old iterator belong to the position we just
        # left, so the iterator goes with it.
        self._decoder = None

    def _frames(self):
        """The live frame iterator, opened on first use after a seek."""
        if self._decoder is None:
            self._decoder = self.container.decode(self.stream)
        return self._decoder

    def _decode_and_check_frames(self, use_time: bool, target_pts: int, idx: int):
        """Decode from stream, recovering once if the decoder is already at EOF."""
        try:
            return self._scan_for_frame(use_time, target_pts, idx)
        except EOFError:
            # The decoder was left at EOF by an earlier read that had to scan the
            # whole stream, and PyAV raises rather than yielding nothing when a
            # packet is sent to it in that state. A seek flushes it; retry once.
            # Not recursive on purpose: exactly one retry, so a genuinely
            # unreadable target still surfaces instead of looping.
            self._seek(target_pts)
            return self._scan_for_frame(use_time, target_pts, idx)

    def _scan_for_frame(self, use_time: bool, target_pts: int, idx: int):
        """Decode forward to ``target_pts``, restarting earlier if it is missed.

        ``container.seek`` is not exact in every container. MPEG program and
        transport streams (``.mpg``, ``.ts``) ignore the ``backward`` flag:
        FFmpeg seeks to the timestamp itself rather than to the keyframe before
        it, and since that position is mid-GOP the first frame the decoder can
        emit belongs to the *next* keyframe -- past what was asked for. The call
        reports success either way, so a late seek can only be recognised from
        what comes back. mp4, mkv and webm honour the flag, so for those the
        first pass always succeeds and nothing below the first line runs.

        Restarting one keyframe further back at a time is the workaround the
        FFmpeg mailing list recommends for these containers: seek behind the
        target, then decode forward to it. The list of restart points is finite,
        so a target that genuinely cannot be reached ends the loop instead of
        retrying forever.
        """
        last_idx, frame, missed = self._scan_once(use_time, target_pts, idx)
        if not missed:
            return last_idx, frame

        rewind_points = self._rewind_points(target_pts)
        for rewind_pts in rewind_points:
            self._seek(rewind_pts)
            # Each restart point sits one tick before a keyframe, so that keyframe
            # is what the decoder must hand back first. Insisting on it is what
            # makes the retry worth doing: the same containers that seek late also
            # mislabel timestamps afterwards, handing back a frame carrying the
            # target's pts but decoded from the wrong reference. Landing anywhere
            # else means this restart point is unusable, not that the frame is.
            last_idx, frame, missed = self._scan_once(
                use_time, target_pts, idx, expect_keyframe_pts=rewind_pts + 1
            )
            if not missed:
                return last_idx, frame

        # Out of restart points. Hand back the closest frame found rather than
        # raising -- but say so, because the answer is not the frame requested
        # and silently returning the wrong one is the failure this guards.
        logger.warning(
            "could not reach pts %s for frame %s in %s after %d restarts; "
            "returning the nearest frame decoded",
            target_pts,
            idx,
            self.file_path.name,
            len(rewind_points),
        )
        return last_idx, frame

    def _publish_decoded(self, frame: av.VideoFrame) -> None:
        """Cache a freshly decoded frame and record it as the read position."""
        self._buffer.put(self._get_frame_idx(frame.pts)[0], frame)
        self._stream_pts = frame.pts

    def _recover_frame(self, use_time: bool, target_pts: int, idx: int):
        """Re-acquire the frame for ``idx`` after the read position went wrong.

        Shares the single-frame path's recovery, then republishes the position so
        a surrounding loop can carry on streaming from where this left off.
        """
        _, frame = self._decode_and_check_frames(use_time, target_pts, idx)
        if frame is not None:
            self._publish_decoded(frame)
        return frame

    def _rewind_points(self, target_pts: int) -> list[int]:
        """Restart timestamps to try, each one keyframe further back.

        One tick *before* each keyframe rather than the keyframe itself: where
        the backward flag is ignored, asking for a keyframe's own timestamp is
        exactly what overshoots it, while asking for the tick before it lands on
        it. The stream start is always the final fallback, since decoding from
        there cannot miss a frame that exists.
        """
        with self._lock:
            earlier = {int(k) for k in self._keyframe_pts if k <= target_pts}
        points = [k - 1 for k in sorted(earlier, reverse=True)][: _MAX_REWINDS - 1]
        start = int(self.stream.start_time or 0) - 1
        if start not in points:
            points.append(start)
        return points

    def _scan_once(
        self,
        use_time: bool,
        target_pts: int,
        idx: int,
        expect_keyframe_pts: int | None = None,
    ):
        """Decode forward from wherever the stream is, looking for ``target_pts``.

        One pass, no seeking: `_scan_for_frame` owns the retrying. Returns the
        frame found and whether the pass *missed* the target because the stream
        was in the wrong place. Three ways that happens, all meaning the read
        position is wrong rather than the frame absent:

        - the first frame the decoder produced was already past the target, so
          there was nothing at or before it to return;
        - the stream ran out before the target was ever reached;
        - ``expect_keyframe_pts`` was given and the decoder did not open on that
          keyframe, so everything decoded after it is referenced against the
          wrong picture even where the timestamps look right.

        Returning the frame just *before* the target is not a miss: that is the
        intended answer when a timestamp falls between two frames.
        """
        preceding_frame = None
        last_idx = self.last_loaded_idx
        frame_duration = 1 / float(self.stream.average_rate)
        time_threshold = self.round_fn(idx * frame_duration)

        for frame in self._frames():
            if frame.pts is None:
                continue
            if expect_keyframe_pts is not None:
                if not frame.key_frame or frame.pts != expect_keyframe_pts:
                    return last_idx, None, True
                expect_keyframe_pts = None
            if (not use_time and frame.pts > target_pts) or (
                use_time and frame.time > time_threshold
            ):
                last_idx = idx
                current_frame = preceding_frame or frame
                return last_idx, current_frame, preceding_frame is None
            elif (not use_time and frame.pts == target_pts) or (
                use_time and frame.time == time_threshold
            ):
                last_idx = idx
                current_frame = frame
                return last_idx, current_frame, False
            preceding_frame = frame
            # Checked only after the frame is accounted for: the iterator has
            # already moved past it, so recording it as the read position (and
            # keeping the iterator) is what lets the next request stream forward
            # from here instead of seeking back to the keyframe.
            if self._abort_decoding is not None and self._abort_decoding():
                self._publish_decoded(frame)
                raise _Superseded()

        # Falling out of the loop means the generator was exhausted, so the
        # container and codec are now at EOF. This happens routinely on B-frame
        # codecs when the target is near the end of the stream, since packets
        # arrive in decode order rather than display order. Record it: the next
        # decode must re-seek first or PyAV raises EOFError.
        self._at_eof = True
        # An exhausted iterator yields nothing forever; drop it so the re-seek
        # opens a fresh one rather than silently finding no frames.
        self._decoder = None
        # Running out before reaching the target is the other shape of a late
        # seek: it puts the read position past the target's own GOP, so the
        # frames that would have matched were never decoded. Distinguish that
        # from legitimately stopping at the last frame of the stream.
        missed = preceding_frame is None or (
            not use_time and preceding_frame.pts < target_pts
        )
        return last_idx, preceding_frame, missed

    @property
    def shape(self) -> tuple[int, ...]:
        """
        :
            Shape of the video, matching the array returned by indexing:
            ``(n_frames, height, width, 3)`` for ``pixel_format="rgb24"``,
            ``(n_frames, height * 3 // 2, width)`` for ``"yuv420p"``, and
            ``(n_frames, 3, height, width)`` for ``"yuv444p"``. When
            ``pixel_format`` is ``None``, frames are returned as
            `av.VideoFrame` and the trailing dimensions are those of
            ``frame.to_ndarray()`` in the stream's native format, measured
            from the first decoded frame.

        Notes
        -----
        - When the total frame count is unknown at initialization, the length
          may grow while the background indexer discovers frames. A warning is
          emitted until indexing is complete.
        """
        if self._n_frames is None:
            self._wait_for_all_pts()
        return self._n_frames, *self._frame_shape

    def to_rgb(self, frames) -> NDArray:
        """
        Convert frames returned by this reader to RGB.

        Same as the module-level `asyncvideo.to_rgb`, except that the source
        format is taken from this reader rather than passed in: whatever
        ``pixel_format`` was requested, or — when that is ``None`` — the
        stream's native format, read off a decoded frame.

        Parameters
        ----------
        frames :
            Any output of `get` or ``__getitem__`` on this reader.

        Returns
        -------
        :
            ``(H, W, 3)`` uint8 for a single frame, ``(n, H, W, 3)`` for a stack.

        Examples
        --------
        >>> vh = VideoHandler("example.mp4", pixel_format="yuv444p")  # doctest: +SKIP
        >>> # (3, H, W) on its own is ambiguous, but the reader knows the format
        >>> rgb = vh.to_rgb(vh[7])  # doctest: +SKIP
        """
        from_format = self.pixel_format
        if from_format is None and self.current_frame is not None:
            # frames are av.VideoFrame here, which to_rgb reads directly; this
            # only matters if the caller already called to_ndarray() themselves
            from_format = self.current_frame.format.name
        return to_rgb(frames, from_format=from_format)

    @property
    def frame_shape(self) -> tuple[int, int]:
        """
        :
            Pixel grid of a single frame, ``(height, width)``.

        Notes
        -----
        - This is always the true frame size, independent of ``pixel_format``.
          It therefore differs from the trailing entries of `shape` for the
          packed layouts: a 480x640 ``yuv420p`` video has ``frame_shape``
          ``(480, 640)`` while ``to_ndarray()`` returns ``(720, 640)``.
        """
        return self.stream.height, self.stream.width

    @property
    def _frame_shape(self) -> tuple[int, ...]:
        """Per-frame array shape for the chosen pixel format."""
        h, w = self.stream.height, self.stream.width
        if self.pixel_format == "rgb24":
            return (h, w, 3)
        elif self.pixel_format == "yuv420p":
            return (h * 3 // 2, w)
        elif self.pixel_format == "yuv444p":
            return (3, h, w)
        elif self.pixel_format is None:
            # No conversion was requested, so ``to_ndarray()`` lays the frame out
            # according to the stream's *native* format (yuv420p, gray, ...), not
            # as (h, w, 3). Rather than mirror PyAV's format table here, measure
            # it once on a real frame and cache the result.
            if self._native_frame_shape is None:
                if self.current_frame is None:
                    # Called before the first decode (``shape`` is used during
                    # __init__ to resolve n_frames). Report the pixel grid as a
                    # provisional answer and leave the cache unset so the
                    # measured shape wins as soon as a frame exists.
                    return (h, w)
                self._native_frame_shape = self.current_frame.to_ndarray().shape
            return self._native_frame_shape
        else:
            raise ValueError(f"Unsupported pixel_format: {self.pixel_format!r}")

    @property
    def time(self) -> NDArray:
        """
        :
            Timestamp of every frame, in seconds, one entry per frame.

        Notes
        -----
        - If a ``time`` array was given at initialization, that array is
          returned. Otherwise the times are the stream's own presentation
          timestamps, so they are the real frame times rather than a uniform grid
          derived from the nominal frame rate -- which matters for variable frame
          rate video, and for recordings that drop frames.
        - These are absolute times on the container's clock, so the first frame is
          not necessarily at ``0``. Subtract ``time[0]`` for times relative to the
          start of the video.
        - Reading this blocks until the background indexer has seen every frame,
          since a frame's timestamp is not known before then. Indexing a video by
          frame number never waits on it.

        Raises
        ------
        ValueError
            If a ``time`` array was given whose length does not match the number
            of frames actually found in the video.
        """
        return self._time_future.result()

    def _wait_for_all_pts(self, timeout=None):
        """Wait until the PTS index thread has finished."""
        self._index_ready.wait(timeout)

    def _wait_for_key_pts(self, timeout=None):
        """Wait until the keyframe PTS thread has finished."""
        self._pts_keyframe_ready.wait(timeout)

    def _wait_for_index(self, timeout=None):
        """Wait until both the PTS index and keyframe threads have finished."""
        self._wait_for_all_pts(timeout)
        self._wait_for_key_pts(timeout)

    def get_slice(self, start: float, end: float | None = None):
        # TODO check start and end are sorted
        start = self._ts_to_index(start, self.time)
        if end:
            end = self._ts_to_index(end, self.time)
            return slice(start, end)
        else:
            return slice(start, start + 1)

    def _append_frame(self, frames, idx, frame):
        if self.pixel_format is not None:
            frames[idx] = frame.to_ndarray(format=self.pixel_format)
        else:
            frames.append(frame)

    def _decode_multiple(
        self,
        idx_start: int,
        idx_end: int,
        step: int = 1,
    ) -> tuple[int, list[av.VideoFrame | NDArray], av.VideoFrame]:
        effective_end = min(idx_end, self.shape[0])
        indices = np.arange(idx_start, effective_end, step)
        num_frames = len(indices)
        time_threshold_all = self.round_fn(indices)

        if self.pixel_format is not None:
            frames = np.empty(
                (num_frames, *self._frame_shape),
                dtype=np.uint8,
            )
        else:
            frames = []

        collected = 0

        # initialize current frame
        if self.current_frame is None:
            self.get(0)

        preceding_frame = self.current_frame
        last_frame = self.current_frame
        # The first target is seeked to unconditionally, as before; from then on
        # the shared iterator carries the position (and any queued frames)
        # forward, including past the end of this call.
        seeked = False

        while collected < num_frames:
            # check buffer first
            cached = self._buffer.get(indices[collected])
            if cached is not None:
                self.current_frame = cached
                self.last_loaded_idx = indices[collected]
                self._append_frame(frames, collected, cached)
                preceding_frame = cached
                last_frame = cached
                collected += 1
                continue

            target_pts, use_time = self._get_target_frame_pts(indices[collected])

            # Seek when the target is behind us or past a nearer keyframe.
            if not seeked or self._need_seek_call(self._stream_pts, target_pts):
                self._seek(target_pts)
                seeked = True
                # Whatever was decoded before the seek describes the position we
                # just left, so it is not a candidate answer for anything after
                # it -- and its absence is what marks a seek that landed late.
                preceding_frame = None

            # Advance one frame. container.decode handles B-frame buffering
            # internally, so zero-frame packets are transparent to us.
            try:
                frame = next(f for f in self._frames() if f.pts is not None)
            except StopIteration:
                self._at_eof = True
                self._decoder = None
                frame = None

            time_threshold = time_threshold_all[collected]
            passed_target = frame is not None and (
                (frame.pts > target_pts)
                if not use_time
                else (frame.time > time_threshold)
            )

            # Same late seek the single-frame path handles: either the stream ran
            # out before the target, or the first frame after the seek was already
            # past it with nothing before it to fall back on. Both mean the read
            # position is wrong rather than the frame missing, so recover this one
            # index the same way and carry on streaming from there.
            if frame is None or (passed_target and preceding_frame is None):
                frame = self._recover_frame(use_time, target_pts, indices[collected])
                if frame is None:
                    break
                self._append_frame(frames, collected, frame)
                collected += 1
                last_frame = preceding_frame = frame
                continue

            self._publish_decoded(frame)

            found_current = (
                (frame.pts == target_pts)
                if not use_time
                else (frame.time == time_threshold)
            )

            if passed_target:
                frame = preceding_frame
                self._append_frame(frames, collected, frame)
                collected += 1
            elif found_current:
                self._append_frame(frames, collected, frame)
                collected += 1

            last_frame = frame
            preceding_frame = frame

        return indices[-1], frames, last_frame

    def __getitem__(
        self,
        idx: int | slice | tuple[int | slice, tuple[slice, ...]],
    ) -> NDArray | av.VideoFrame | list[av.VideoFrame]:
        """
        Get item for video frame.

        Gets one or more frames from a video.

        Parameters
        ----------
        idx:
            The index for slicing. Can be:

            - ``int``: a single frame index.
            - ``slice``: a range of frame indices.
            - ``tuple[int | slice, *tuple[slice, ...]]``: the first element
              selects frames; the remaining elements are optional spatial
              slices ``(height, width, channel)`` applied to the decoded
              array after decoding. Decoding always uses the full spatial
              extent; the spatial slices are cheap numpy views.

        Returns
        -------
        ndarray or av.VideoFrame or list[av.VideoFrame]
            - ``int`` → single frame ``(height, width, 3)``.
            - ``slice`` → ``(n_frames, height, width, 3)`` array or
              ``list[av.VideoFrame]``.
            - ``tuple`` → same as above with spatial slices applied.
        """
        # Unpack tuple: first element is time, rest are spatial slices
        spatial_idx: tuple[slice, ...] | None = None
        if isinstance(idx, tuple):
            spatial_idx = idx[1:] if len(idx) > 1 else None
            idx = idx[0]

        time_is_int = isinstance(idx, int)

        if isinstance(idx, slice):
            # Resolve frame count once — fast path if already known, otherwise waits.
            n_frames = self._n_frames if self._n_frames is not None else self.shape[0]

            # Fill in missing slice components
            start = idx.start or 0
            if start >= n_frames:
                if self.pixel_format is not None:
                    return np.empty((0, *self._frame_shape), dtype=np.uint8)
                else:
                    return []
            stop = idx.stop if idx.stop is not None else n_frames
            step = idx.step if idx.step is not None else 1

            # convert negative vals
            start = start if start >= 0 else start + n_frames
            start = max(0, min(start, n_frames))
            stop = stop + n_frames if stop < 0 else stop
            stop = max(0, min(stop, n_frames))

            # revert slice if negative step
            revert = step < 0
            step = abs(step)

            if (stop - start) // step > 1:
                target_pts, _ = self._get_target_frame_pts(start)

                if self._stream_pts is None or self._need_seek_call(
                    self._stream_pts, target_pts
                ):
                    self._seek(target_pts)

                frame_idx, frames, last_frame = self._decode_multiple(
                    start, stop, step=step
                )
                # update current decoded frame
                if len(frames):
                    self.last_loaded_idx = frame_idx
                    self.current_frame = last_frame
                    self._stream_pts = last_frame.pts
                frames = frames if not revert else frames[::-1]
                if spatial_idx is not None and isinstance(frames, np.ndarray):
                    frames = frames[(slice(None), *spatial_idx)]
                return frames

        # Default case: single index
        # TODO Check borders
        idx_start = idx if not hasattr(idx, "start") else idx.start
        n_frames = self._n_frames if self._n_frames is not None else self.shape[0]
        idx_start = idx_start if idx_start >= 0 else n_frames + idx_start
        frame = self._get_by_index(idx_start)
        # handle slice requesting a single frame:
        # for arrays add 1 dimension (1, pixel, pixel)
        # for frames return a len 1 list.
        if isinstance(idx, slice):
            if isinstance(frame, np.ndarray):
                frame = np.expand_dims(frame, axis=0)
            else:
                frame = [frame]

        # Apply spatial slices (height, width, channel) after decoding.
        # For a single frame (H, W, 3) index directly; for a stack
        # (N, H, W, 3) prepend a full-axis slice to keep the time dimension.
        if spatial_idx is not None and isinstance(frame, np.ndarray):
            if time_is_int:
                frame = frame[spatial_idx]
            else:
                frame = frame[(slice(None), *spatial_idx)]

        return frame

    def __len__(self):
        return self.shape[0]

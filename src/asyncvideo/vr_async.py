from __future__ import annotations

import multiprocessing
import queue as _stdlib_queue
import sys
import threading
from concurrent.futures import Future
from multiprocessing import Queue
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._pyav_video_reader import VideoHandler
from ._vr_process import _reader_process
from .convert import to_rgb
from .utils import (
    Colorspace,
    FutureArray,
    ReaderError,
    SharedMemRGB,
    SharedMemYUV,
    create_buffers,
    create_shared_memory,
)

# fork clones the parent process directly — no re-import of the main script,
# so AsyncVideoReader can be instantiated at module level (e.g. in scripts or
# IPython) without hitting the spawn bootstrap trap. Windows only has spawn.
if sys.platform == "win32":
    mp_ctx = multiprocessing.get_context("spawn")
else:
    mp_ctx = multiprocessing.get_context("fork")


class AsyncVideoReader:
    """
    Video reader that decodes in a separate process.

    Requesting a frame returns a `concurrent.futures.Future` immediately, and the
    decode happens in a worker process, so the calling thread is never blocked by
    it. Each reader owns one process, so several readers decode genuinely in
    parallel — which is the point when displaying more than one video at a time.

    Frames come back in the video's native pixel format, one per request. Use
    `to_rgb` to convert them for display.

    A new request supersedes any older one still in flight: the earlier future is
    cancelled rather than queued. Indexing on every change of a slider therefore
    stays responsive instead of working through frames that are no longer wanted.

    Parameters
    ----------
    path :
        Path to the video file.
    time :
        Timestamps for each frame, in seconds. Pass this when frame times come
        from an acquisition system rather than a constant frame rate; `get` and
        `t` then use this clock. If ``None``, a uniform grid is derived from the
        stream's average rate.
    yuv_packed :
        For a ``yuv420p`` video, transfer the frame as a single packed
        ``(1, H * 3 // 2, W)`` array rather than as a ``(Y, U, V)`` tuple of
        planes. Ignored for ``rgb24`` video.
    stream_index :
        Index of the video stream to read, for files carrying more than one.
    buffer_size :
        Number of recently decoded frames the worker keeps cached. A request that
        hits the cache needs no seeking or decoding. Default is 30.

    Notes
    -----
    - The reader owns a process, so call `shutdown` when finished with it.
    - There is no ``pixel_format`` parameter, unlike `VideoHandler`: frames are
      always decoded in the video's native format. Use `to_rgb` to convert them.

    Examples
    --------
    >>> from asyncvideo import AsyncVideoReader
    >>> reader = AsyncVideoReader("example.mp4")  # doctest: +SKIP
    >>> future = reader[4200]  # returns at once  # doctest: +SKIP
    >>> frame = reader.to_rgb(future.result())[0]  # doctest: +SKIP
    >>> reader.shutdown()  # doctest: +SKIP

    See Also
    --------
    VideoHandler
        Synchronous reader. Decodes in the calling thread, and can return a range
        of frames from one request.
    """

    def __array__(self) -> AsyncVideoReader:
        return self

    def __init__(
        self,
        path: str | Path,
        time: NDArray | None = None,
        yuv_packed: bool = False,
        stream_index: int = 0,
        buffer_size: int = 30,
    ) -> None:
        self._path = Path(path)

        # pixel_format is deliberately fixed: frames cross the process boundary in
        # the stream's native layout, since the shared-memory segments are sized
        # from it. Everything else is forwarded to the worker's handler so that
        # time lookups there use the same clock as on this side.
        self._handler_kwargs = {
            "stream_index": stream_index,
            "time": time,
            "buffer_size": buffer_size,
        }

        vr = VideoHandler(self._path, pixel_format=None, **self._handler_kwargs)
        frame0 = vr[(slice(0, 1),)][0]

        # Frame times are published once by the worker, whose handler owns them,
        # and cached here on first read. Deliberately not taken from ``vr`` above:
        # its times are only known after a full index pass, and waiting for that
        # would slow construction for every caller, including those who only ever
        # index by frame number.
        self._time: NDArray | None = None

        # take the pixel grid from the frame itself: VideoHandler.shape reports the
        # native to_ndarray() layout here, which for yuv420p is (h * 3 // 2, w)
        self._shape_frame = frame0.height, frame0.width
        n_frames = vr.shape[0]

        colorspace = Colorspace(frame0.format.name)

        self._colorspace = colorspace
        frame0_numpy = frame0.to_ndarray()
        self._dtype = frame0_numpy.dtype

        if self.colorspace == Colorspace.rgb24:
            self._shape = (n_frames, *self._shape_frame, 3)
            self._shape_chroma = None

        elif self.colorspace == Colorspace.yuv420p:
            self._shape = (n_frames, *self._shape_frame)
            self._shape_chroma = (
                frame0.format.chroma_height(),
                frame0.format.chroma_width(),
            )

        n_frames = 1

        self._yuv_packed = yuv_packed

        self._shared_mems = create_shared_memory(
            frame0, n_frames=n_frames, yuv_packed=self._yuv_packed
        )
        shared_mem_names = tuple(b.name for b in self.shared_mems)

        vr.close()
        del vr

        self._request_queue: Queue = mp_ctx.Queue()
        self._response_queue: Queue = mp_ctx.Queue()
        # carries exactly one message: the frame times, or the error raised while
        # resolving them. Kept off the request queue, which is drained of stale
        # entries on every new request and would discard it.
        self._time_queue: Queue = mp_ctx.Queue()

        self._stop_event = mp_ctx.Event()
        self._buffer_lock = mp_ctx.Lock()

        self._pending_rid: int = 0
        # shared with the worker process. The worker uses this to skip any
        # request whose rid has already been superseded by a newer one,
        # without false-dropping the current request (a single ``cancel_event``
        # bit can't distinguish *which* request was cancelled).
        self._latest_rid = mp_ctx.Value("q", 0)
        self._pending_future: FutureArray | None = None
        # what the request in flight asked for, so that a request for the same frame can share
        # its decode instead of superseding it
        self._pending_selector: tuple | None = None
        # the futures sharing the decode of the request in flight
        self._shared_futures: list[FutureArray] = []
        self._listener_lock = threading.Lock()

        # guards the shared-memory teardown so a second shutdown() cannot
        # unlink segments that are already gone
        self._release_lock = threading.Lock()
        self._released = False

        self._buffer = create_buffers(
            self._shared_mems,
            colorspace=self.colorspace,
            shape_frame=self._shape_frame,
            shape_chroma=self._shape_chroma,
            n_frames=1,
            yuv_packed=self._yuv_packed,
        )

        self._worker = mp_ctx.Process(
            target=_reader_process,
            kwargs={
                "path": self._path,
                "shared_mem_names": shared_mem_names,
                "colorspace": self.colorspace,
                "shape_frame": self._shape_frame,
                "shape_chroma": self._shape_chroma,
                "yuv_packed": yuv_packed,
                "handler_kwargs": self._handler_kwargs,
                "time_queue": self._time_queue,
                "request_queue": self._request_queue,
                "response_queue": self._response_queue,
                "stop_event": self._stop_event,
                "latest_rid": self._latest_rid,
                "buffer_lock": self._buffer_lock,
            },
            daemon=True,
        )
        self._worker.start()

        self._listener = threading.Thread(target=self._listen, daemon=True)
        self._listener.start()

    @property
    def shared_mems(self) -> SharedMemRGB | SharedMemYUV:
        return self._shared_mems

    @property
    def colorspace(self) -> Colorspace:
        return self._colorspace

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def to_rgb(self, frames) -> np.ndarray:
        """
        Convert a resolved request from this reader to RGB.

        Same as the module-level `asyncvideo.to_rgb`, except that the source
        format is taken from this reader's ``colorspace``.

        Parameters
        ----------
        frames :
            The result of a future returned by ``__getitem__`` — either a
            ``(Y, U, V)`` plane tuple or, with ``yuv_packed=True``, a packed
            array.

        Returns
        -------
        :
            ``(n, H, W, 3)`` uint8.

        Examples
        --------
        >>> reader = AsyncVideoReader("example.mp4")  # doctest: +SKIP
        >>> rgb = reader.to_rgb(reader[(10,)].result())  # doctest: +SKIP
        """
        return to_rgb(frames, from_format=str(self.colorspace))

    @property
    def time(self) -> NDArray:
        """
        :
            Timestamp of every frame, in seconds, one entry per frame.

        Notes
        -----
        - The ``time`` array given at construction, or the stream's own
          presentation timestamps. See `VideoHandler.time`.
        - Reading this the first time blocks until the worker has indexed every
          frame, since the timestamps are not known before then. `get` does not
          wait on it: the timestamp is resolved in the worker, which owns the
          times already.

        Raises
        ------
        ValueError
            If a ``time`` array was given whose length does not match the number
            of frames actually found in the video.
        """
        if self._time is None:
            kind, payload = self._time_queue.get()
            if kind == "error":
                raise payload
            self._time = payload
        return self._time

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def _listen(self):
        while True:
            msg = self._response_queue.get()
            if msg is None:
                break

            rid, status = msg

            with self._listener_lock:
                if rid != self._pending_rid:
                    continue

                if status != ReaderError.ok:
                    # Fail the futures rather than leave the consumers blocked: the
                    # worker survives the error, so nothing else will ever
                    # resolve this request. The traceback is in the worker log.
                    error = RuntimeError(
                        f"reader process failed to decode request {rid} "
                        f"({ReaderError(status).name})"
                    )
                    for future in (self._pending_future, *self._shared_futures):
                        future.set_exception(error)
                    self._shared_futures.clear()
                    continue

                # TODO: if shared mem changes due to different number of frames
                # if shm_name != self._shared_mems.name:
                #     self._shared_mems.unlink()
                #     self._shared_mems.close()
                #     self._shared_mems = SharedMemory(name=shm_name)
                #
                #     self._result = np.ndarray(
                #         frame_shape, dtype=np.dtype(dtype), buffer=self._shared_mems.buf
                #     )

                futures = (self._pending_future, *self._shared_futures)
                self._shared_futures.clear()

                # set_result must happen while holding _listener_lock, otherwise
                # __getitem__ can cancel a future between this point and the
                # set_result call below, raising InvalidStateError and killing
                # the listener thread (the reader then permanently hangs).
                with self._buffer_lock:
                    # every consumer gets its own copy, none of them can modify
                    # the frame another one was given
                    for future in futures:
                        if self.colorspace == Colorspace.rgb24 or self._yuv_packed:
                            future.set_result(self._buffer.copy())

                        elif self.colorspace == Colorspace.yuv420p:
                            future.set_result(
                                (
                                    self._buffer[0].copy(),
                                    self._buffer[1].copy(),
                                    self._buffer[2].copy(),
                                )
                            )

    @staticmethod
    def _frame_index(index):
        """Extract the frame selector from an index.

        Accepts ``reader[i]`` and ``reader[i:j]`` as well as the tuple form
        ``reader[(i,)]``, where only the first entry selects frames and any
        remaining entries are spatial slices handled downstream.

        Unpacking happens before ``__getitem__`` touches any state: it cancels
        the previous request and bumps the request id, so an index that cannot
        even be unpacked must fail before that, not halfway through.
        """
        if isinstance(index, tuple):
            if not index:
                raise IndexError("an empty tuple is not a valid frame index")
            index = index[0]
        if isinstance(index, np.integer):
            return int(index)
        # anything else (including a plain int or slice) is forwarded as-is; the
        # worker reports an unservable index by failing the future
        return index

    def __getitem__(self, index) -> FutureArray:
        """
        Request the frame at a given index, without waiting for it.

        Parameters
        ----------
        index :
            Frame index. Accepts ``reader[i]``, ``reader[i:j]`` and the tuple form
            ``reader[(i,)]``, whose first entry selects the frame. Since one frame
            is transferred per request, a slice still resolves to a single frame.

        Returns
        -------
        :
            A future resolving to the frame in the video's native format: a
            ``(Y, U, V)`` tuple of planes for ``yuv420p``, or a single array for
            ``rgb24`` and for ``yuv_packed=True``. Each carries a leading axis of
            length 1. Cancels any request still in flight.
        """
        return self._submit(self._frame_index(index), by_time=False)

    def get(self, ts: float) -> FutureArray:
        """
        Request the frame at a timestamp, without waiting for it.

        The timestamp is resolved against `t`, so this is the counterpart of
        `VideoHandler.get` and is the reliable way to address a moment rather than
        a position: cameras recorded together often run at different frame rates,
        so the same instant is a different frame index in each video.

        Parameters
        ----------
        ts :
            Time in seconds, on the same clock as `t`.

        Returns
        -------
        :
            A future resolving to the frame at, or immediately before, ``ts``, in
            the same form `__getitem__` returns. Cancels any request still in
            flight.

        Examples
        --------
        >>> reader = AsyncVideoReader("example.mp4")  # doctest: +SKIP
        >>> frame = reader.to_rgb(reader.get(12.5).result())[0]  # doctest: +SKIP
        """
        return self._submit(float(ts), by_time=True)

    def _submit(self, selector, by_time: bool) -> FutureArray:
        """
        Queue one request, superseding whatever is still in flight.

        A request for the frame already in flight shares that decode instead of
        superseding it. Without this, several consumers reading the same frame from one
        reader would cancel each other and only the last one would be served, e.g. one
        video displayed in several subplots.
        """
        with self._listener_lock:
            if (
                self._pending_future is not None
                and not self._pending_future.done()
                and self._pending_selector == (selector, by_time)
            ):
                shared = Future()
                self._shared_futures.append(shared)
                return shared

            if self._pending_future is not None and not self._pending_future.done():
                self._pending_future.cancel()
            for shared in self._shared_futures:
                shared.cancel()
            self._shared_futures.clear()
            self._pending_selector = (selector, by_time)

            self._pending_rid += 1
            # publish to the worker so it knows the newest rid in flight
            self._latest_rid.value = self._pending_rid
            future = Future()
            self._pending_future = future

        # drain stale entries before enqueuing the new one. The worker will
        # skip them via the rid check anyway, but each multiprocessing.Queue
        # ``get`` is several ms of cross-process IPC - hundreds of stale
        # entries (from rapid slider dragging) turn into multi-second delays
        # before the worker reaches the latest request.
        while True:
            try:
                self._request_queue.get_nowait()
            except _stdlib_queue.Empty:
                break

        self._request_queue.put((self._pending_rid, selector, by_time))
        return future

    def shutdown(self, wait: bool = True):
        self._stop_event.set()
        self._request_queue.put(None)  # wake up the worker if blocked on get()
        if wait:
            self._release()
        else:
            # The teardown cannot run inline here: ``_buffer`` is a numpy view
            # onto the shared segments and ``_listen`` may still be copying out
            # of it, so unmapping them while it runs is a use-after-free (a
            # segfault, not an exception). Hand the join + teardown to a helper
            # thread so the caller still returns immediately.
            threading.Thread(target=self._release, daemon=True).start()

    def _release(self):
        """Stop the worker and listener, then unmap and destroy the segments."""
        self._worker.join()
        # only stop the listener once the worker is gone and no further results
        # can land on the queue
        self._response_queue.put(None)
        self._listener.join()

        # Held across the whole teardown, not just the flag check: a concurrent
        # shutdown() must block until the segments are actually gone rather than
        # return early on a flag that is set but not yet acted on. (An Event is
        # the wrong primitive here — is_set()/set() is a check-then-act, so two
        # callers can both reach unlink() and the second raises FileNotFoundError.)
        with self._release_lock:
            if self._released:
                return
            self._released = True

            # drop the numpy views before unmapping the memory they point at
            self._buffer = None

            # _shared_mems is a tuple: 1 segment for rgb24/packed-yuv, 3 for
            # planar yuv. close() releases this process's mapping; unlink()
            # destroys the segment and must happen exactly once, from the owner
            # — this process created them, so it unlinks and the worker only
            # closes.
            for shm in self._shared_mems:
                shm.close()
                shm.unlink()

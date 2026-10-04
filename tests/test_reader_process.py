"""The worker loop of :class:`AsyncVideoReader`, driven directly.

``_reader_process`` runs here on a thread rather than in a child process, so a
test can reach into it -- in particular, supersede a request at an exact point
partway through its decode, which a real reader can only hit by timing.
"""

import logging
import multiprocessing as mp
import queue
import threading

import numpy as np
import pytest

from asyncvideo import _vr_process
from asyncvideo._pyav_video_reader import VideoHandler
from asyncvideo._vr_process import _reader_process, _RequestGuard
from asyncvideo.utils import Colorspace, ReaderError, create_shared_memory

RESULT_TIMEOUT = 30.0


def test_request_guard_tracks_the_request_being_served():
    latest_rid = mp.Value("q", 0)
    guard = _RequestGuard(latest_rid)

    guard.rid = latest_rid.value = 1
    assert not guard.superseded()

    # the parent submits a newer request while this one is being served
    latest_rid.value = 2
    assert guard.superseded()

    # the worker moves on to it
    guard.rid = 2
    assert not guard.superseded()


@pytest.fixture()
def worker(video_path, monkeypatch):
    """Start ``_reader_process`` on a thread, with a hook to supersede mid-decode.

    ``supersede_after`` is a dict the test fills in before submitting: on the
    worker's ``checks``-th abort check, ``latest_rid`` is bumped to ``to`` --
    exactly what the parent does when a newer request is submitted. It fires
    once, so the request served after it runs undisturbed.
    """
    latest_rid = mp.Value("q", 0)
    supersede_after: dict = {}
    handlers: list[VideoHandler] = []

    class _SupersededMidDecode(VideoHandler):
        def __init__(self, *args, abort_decoding, **kwargs):
            checks = 0

            def abort() -> bool:
                nonlocal checks
                if supersede_after:
                    checks += 1
                    if checks == supersede_after["checks"]:
                        latest_rid.value = supersede_after["to"]
                return abort_decoding()

            super().__init__(*args, abort_decoding=abort, **kwargs)
            handlers.append(self)

    monkeypatch.setattr(_vr_process, "VideoHandler", _SupersededMidDecode)

    with VideoHandler(video_path) as probe:
        frame0 = probe[0]
    shared_mems = create_shared_memory(frame0, n_frames=1, yuv_packed=True)

    request_queue = mp.Queue()
    response_queue = mp.Queue()
    stop_event = mp.Event()
    buffer_lock = mp.Lock()
    thread = threading.Thread(
        target=_reader_process,
        kwargs={
            "path": video_path,
            "shared_mem_names": tuple(shm.name for shm in shared_mems),
            "colorspace": Colorspace(frame0.format.name),
            "shape_frame": (frame0.height, frame0.width),
            "shape_chroma": None,
            "yuv_packed": True,
            "handler_kwargs": {},
            "time_queue": mp.Queue(),
            "request_queue": request_queue,
            "response_queue": response_queue,
            "stop_event": stop_event,
            "latest_rid": latest_rid,
            "buffer_lock": buffer_lock,
        },
        daemon=True,
    )
    thread.start()

    def frame_in_shared_memory():
        with buffer_lock:
            return np.ndarray(
                (frame0.height * 3 // 2, frame0.width),
                dtype=np.uint8,
                buffer=shared_mems[0].buf,
            ).copy()

    try:
        yield {
            "latest_rid": latest_rid,
            "supersede_after": supersede_after,
            "handlers": handlers,
            "requests": request_queue,
            "responses": response_queue,
            "frame": frame_in_shared_memory,
        }
    finally:
        stop_event.set()
        request_queue.put(None)
        thread.join(timeout=RESULT_TIMEOUT)
        assert not thread.is_alive(), "worker did not stop"
        for shm in shared_mems:
            shm.close()
            shm.unlink()


def test_superseded_decode_moves_on_to_the_next_request(worker, reference, caplog):
    """Aborting mid-decode drops that request silently and serves the next one.

    Request 1 is superseded partway through its scan, with request 2 already
    queued behind it. The worker must answer only request 2 -- no result and no
    error for request 1 -- and the frame it leaves in shared memory must be
    request 2's.
    """
    frames, _ = reference
    caplog.set_level(logging.ERROR, logger=_vr_process.__name__)

    worker["supersede_after"].update(checks=3, to=2)
    worker["latest_rid"].value = 1
    worker["requests"].put((1, 60, False))
    worker["requests"].put((2, 10, False))

    assert worker["responses"].get(timeout=RESULT_TIMEOUT) == (2, ReaderError.ok)
    np.testing.assert_array_equal(worker["frame"](), frames[10])

    with pytest.raises(queue.Empty):
        worker["responses"].get(timeout=0.5)
    assert not caplog.records, "an abort must not be reported as a failure"

    # request 1 really was abandoned partway, not served and then discarded
    (handler,) = worker["handlers"]
    assert 60 not in handler._buffer

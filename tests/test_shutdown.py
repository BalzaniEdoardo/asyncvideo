"""Tests for the bounded, non-blocking teardown of :class:`AsyncVideoReader`.

Ported from pynaviz, where ``PlotVideo.close`` joined its worker inline with a
2 s timeout. A worker created moments earlier has not reached its serve loop
yet -- under the "spawn" start method it is still importing its dependencies --
so it cannot observe the stop event, and the join burned its full timeout on
the GUI thread every time.

Here ``shutdown`` returns immediately by default: the join and the release
of the shared memory run on a tracked helper thread, every wait is bounded,
and readers still live at interpreter exit are shut down by an exit hook that
waits for them.
"""

import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory

import pytest

from asyncvideo import AsyncVideoReader, vr_async
from asyncvideo._vr_process import _reader_process

# ``shutdown`` only signals and starts a thread. Anything near the old 2 s
# join means the regression is back.
MAX_SHUTDOWN_SECONDS = 0.5

# A stub worker lifetime deliberately longer than the old 2 s join timeout, so
# a blocking teardown cannot possibly pass the timing assertions below.
SLOW_WORKER_SECONDS = 3.0

RESULT_TIMEOUT = 30.0
RELEASE_TIMEOUT = 60.0


@pytest.fixture(autouse=True)
def quiet_release_threads():
    """Keep the module-global teardown state from leaking between tests."""
    assert vr_async._drain_releases(timeout=RELEASE_TIMEOUT), "busy before test"
    yield
    assert vr_async._drain_releases(timeout=RELEASE_TIMEOUT), "busy after test"


@pytest.fixture(params=["fork", "spawn"])
def start_method(request, monkeypatch):
    """Run against every start method available on this platform.

    Windows only has "spawn", and that is where the blocking join hurt most, so
    it is covered explicitly rather than inherited from the module default.
    """
    if request.param not in mp.get_all_start_methods():
        pytest.skip(f"{request.param} start method unavailable")
    monkeypatch.setattr(vr_async, "mp_ctx", mp.get_context(request.param))
    return request.param


@pytest.fixture()
def reader(video_path):
    r = AsyncVideoReader(video_path)
    try:
        yield r
    finally:
        r.shutdown(wait=True)


def _segment_names(reader) -> tuple[str, ...]:
    return tuple(shm.name for shm in reader.shared_mems)


def _segment_exists(name: str) -> bool:
    """True if a segment by this name can still be attached.

    On POSIX a successful attach also registers the segment with this process's
    resource_tracker, which would then try to unlink it again at exit, so the
    registration is undone straight away.
    """
    try:
        shm = SharedMemory(name=name)
    except FileNotFoundError:
        return False
    if os.name == "posix":
        resource_tracker.unregister(shm._name, "shared_memory")
    shm.close()
    return True


def _swap_in_stubborn_worker(reader, seconds: float):
    """Replace the reader's worker by one that ignores the stop event.

    Stands in for a worker still importing under spawn, without depending on
    how long imports take on this machine. The real worker is stopped first.
    """
    real = reader._worker
    reader._stop_event.set()
    reader._request_queue.put(None)
    real.join(timeout=RELEASE_TIMEOUT)
    assert not real.is_alive(), "real worker did not stop"

    # time.sleep pickles by reference, so this works under spawn as well
    stub = vr_async.mp_ctx.Process(target=time.sleep, args=(seconds,), daemon=True)
    stub.start()
    reader._worker = stub
    return stub


# ----------------------------------------------------------------------
# The hand-off, isolated from worker start-up timing
# ----------------------------------------------------------------------


def test_shutdown_does_not_wait_for_a_worker_that_ignores_the_stop_event(reader):
    """pynaviz#120, deterministically: a worker outliving the old timeout.

    Called with no arguments, as a GUI closing a video would.
    """
    worker = _swap_in_stubborn_worker(reader, SLOW_WORKER_SECONDS)
    names = _segment_names(reader)

    start = time.perf_counter()
    reader.shutdown()
    elapsed = time.perf_counter() - start

    assert elapsed < MAX_SHUTDOWN_SECONDS, f"shutdown blocked for {elapsed:.2f}s"
    # Returned while the worker is still running: that is the whole point.
    assert worker.is_alive()
    assert all(_segment_exists(n) for n in names), "released before worker exited"

    # The helper thread, not the caller, completes the teardown.
    assert vr_async._drain_releases(timeout=RELEASE_TIMEOUT)
    assert not worker.is_alive()
    assert worker.exitcode == 0, "worker was killed instead of joined"
    assert not any(_segment_exists(n) for n in names)


def test_release_thread_is_a_tracked_daemon(reader):
    """The helper must never keep the interpreter alive, and exit must see it."""
    _swap_in_stubborn_worker(reader, 1.0)
    reader.shutdown()

    with vr_async._release_threads_lock:
        threads = list(vr_async._release_threads)
    assert threads, "helper thread not tracked, exit could not wait for it"
    assert all(t.daemon for t in threads)
    assert all(t.name == "asyncvideo-release" for t in threads)


def test_release_frees_memory_when_the_worker_overruns_its_join(
    reader, monkeypatch, caplog
):
    """A worker that will not exit must not strand the memory forever.

    It is left running rather than terminated: on Windows terminating it was
    seen to free the mapping out from under views still held (pynaviz#120).
    """
    monkeypatch.setattr(vr_async, "_WORKER_JOIN_TIMEOUT", 0.2)
    worker = _swap_in_stubborn_worker(reader, SLOW_WORKER_SECONDS)
    names = _segment_names(reader)

    with caplog.at_level(logging.WARNING, logger=vr_async.__name__):
        reader.shutdown(wait=True)

    assert "releasing shared memory anyway" in caplog.text
    assert worker.is_alive(), "worker was killed"
    assert not any(_segment_exists(n) for n in names)

    worker.join(timeout=RELEASE_TIMEOUT)
    assert worker.exitcode == 0


class _StuckThread:
    """A listener that refuses to stop."""

    def is_alive(self):
        return True

    def join(self, timeout=None):
        return None


def test_release_leaks_memory_rather_than_unmapping_under_a_live_listener(
    reader, caplog
):
    """Unmapping memory a live thread still reads would crash the process."""
    names = _segment_names(reader)
    real_listener = reader._listener
    reader._listener = _StuckThread()

    with caplog.at_level(logging.WARNING, logger=vr_async.__name__):
        reader.shutdown(wait=True)

    assert "did not stop; leaking shared memory" in caplog.text
    assert not reader._released
    assert all(_segment_exists(n) for n in names), "unmapped under a live thread"

    # the sentinel still reached the real listener; a retry now releases
    real_listener.join(timeout=RELEASE_TIMEOUT)
    reader._listener = real_listener
    reader.shutdown(wait=True)
    assert not any(_segment_exists(n) for n in names)


class _RaisingShm:
    """A segment that cannot be released."""

    name = "unreleasable"

    def close(self):
        raise BufferError("cannot close exported pointer")

    def unlink(self):
        raise AssertionError("must not be reached")


def test_release_reports_a_failing_segment_and_still_frees_the_others(
    reader, caplog
):
    """One unreleasable segment must not strand the rest."""
    names = _segment_names(reader)
    reader._shared_mems = (_RaisingShm(), *reader._shared_mems)

    with caplog.at_level(logging.ERROR, logger=vr_async.__name__):
        reader.shutdown(wait=True)

    assert "Unable to release shared memory unreleasable" in caplog.text
    assert not any(_segment_exists(n) for n in names)


def test_deferred_then_blocking_shutdown_releases_exactly_once(reader):
    """A blocking shutdown racing a deferred one must not re-unlink."""
    names = _segment_names(reader)
    reader.shutdown()
    reader.shutdown(wait=True)  # must neither raise nor return before release
    assert reader._released
    assert not any(_segment_exists(n) for n in names)


def test_worker_leaves_before_opening_the_video_when_already_stopped():
    """A worker stopped before it starts serving must not do the work at all.

    The path does not exist: reaching ``VideoHandler`` would raise. ``time`` is
    still answered so a parent blocked on it is released.
    """
    stop_event = mp.Event()
    stop_event.set()
    time_queue = mp.Queue()

    _reader_process(
        path="does-not-exist.mp4",
        shared_mem_names=("does-not-exist",),
        colorspace=None,
        shape_frame=None,
        shape_chroma=None,
        yuv_packed=False,
        handler_kwargs={},
        time_queue=time_queue,
        request_queue=None,
        response_queue=None,
        stop_event=stop_event,
        latest_rid=None,
        buffer_lock=None,
    )

    kind, payload = time_queue.get(timeout=RESULT_TIMEOUT)
    assert kind == "error"
    assert isinstance(payload, RuntimeError)


# ----------------------------------------------------------------------
# End to end, through a real reader under each start method
# ----------------------------------------------------------------------


def test_worker_and_shared_memory_are_created(video_path, start_method):
    """Guard the setup the teardown tests depend on."""
    r = AsyncVideoReader(video_path)
    try:
        assert r._worker._start_method == start_method
        assert r._worker.is_alive()
        assert r._listener.is_alive()
        assert all(_segment_exists(n) for n in _segment_names(r))
        assert r in vr_async._live_readers
    finally:
        r.shutdown(wait=True)
    assert r not in vr_async._live_readers


def test_shutdown_immediately_after_construction_returns_promptly(
    video_path, start_method
):
    """The worst case: no chance at all for the worker to reach its loop."""
    r = AsyncVideoReader(video_path)

    start = time.perf_counter()
    r.shutdown()
    elapsed = time.perf_counter() - start

    assert elapsed < MAX_SHUTDOWN_SECONDS, f"shutdown blocked for {elapsed:.2f}s"
    assert vr_async._drain_releases(timeout=RELEASE_TIMEOUT)
    assert r._released


def test_shutdown_tears_down_worker_listener_and_memory(video_path, start_method):
    """After the helper runs, nothing from the reader is left behind."""
    r = AsyncVideoReader(video_path)
    worker, listener = r._worker, r._listener
    names = _segment_names(r)
    time.sleep(0.2)  # close while the worker is still coming up

    r.shutdown()
    assert r._stop_event.is_set()
    assert vr_async._drain_releases(timeout=RELEASE_TIMEOUT)

    assert not worker.is_alive(), "worker was not joined"
    assert worker.exitcode is not None, "worker was not reaped"
    assert not listener.is_alive(), "listener was not joined"
    assert not any(_segment_exists(n) for n in names)


def test_repeated_open_shutdown_cycles(video_path, start_method):
    """Repeated cycles stay fast and leak neither processes nor memory."""
    workers, names = [], []
    for _ in range(3):
        r = AsyncVideoReader(video_path)
        workers.append(r._worker)
        names += _segment_names(r)
        time.sleep(0.1)

        start = time.perf_counter()
        r.shutdown()
        elapsed = time.perf_counter() - start
        assert elapsed < MAX_SHUTDOWN_SECONDS, f"shutdown blocked for {elapsed:.2f}s"

    assert vr_async._drain_releases(timeout=RELEASE_TIMEOUT)
    assert not any(w.is_alive() for w in workers)
    assert not any(_segment_exists(n) for n in names)


def test_time_after_an_early_shutdown_does_not_hang(video_path, start_method):
    """``time`` is answered whether or not the worker got to open the video."""
    r = AsyncVideoReader(video_path)
    r.shutdown(wait=True)

    outcome = []

    def read_time():
        try:
            outcome.append(len(r.time))
        except RuntimeError as exc:
            outcome.append(exc)

    t = threading.Thread(target=read_time, daemon=True)
    t.start()
    t.join(timeout=RESULT_TIMEOUT)
    assert not t.is_alive(), "reading time after shutdown hung"
    assert outcome


def test_time_raises_when_the_worker_died_without_publishing(reader):
    """A worker gone with nothing on the queue must not leave ``time`` waiting.

    The worker can exit before its publishing thread gets the times out, e.g.
    when shut down right after indexing started.
    """
    # stop the real worker, then drop whatever it may have published
    reader.shutdown(wait=True)
    while True:
        try:
            reader._time_queue.get(timeout=0.5)
        except queue.Empty:
            break

    start = time.perf_counter()
    with pytest.raises(RuntimeError, match="exited before publishing"):
        _ = reader.time
    assert time.perf_counter() - start < 2.0


# ----------------------------------------------------------------------
# The exit hook
# ----------------------------------------------------------------------


def test_exit_hook_shuts_down_live_readers_and_drains(video_path, start_method):
    """Readers never shut down are closed at exit, and exit waits for them."""
    r = AsyncVideoReader(video_path)
    worker = r._worker
    names = _segment_names(r)
    assert r in vr_async._live_readers

    vr_async._shutdown_all_readers()

    assert r._released
    assert not worker.is_alive(), "hook returned with the worker still alive"
    assert not any(_segment_exists(n) for n in names)
    assert not list(vr_async._live_readers)


def test_exit_hook_reports_a_failing_reader_and_continues(
    video_path, monkeypatch, caplog
):
    """One broken reader must not stop the hook closing the rest."""

    class _BrokenReader:
        def shutdown(self, wait=True):
            raise RuntimeError("shutdown failed")

    broken = _BrokenReader()
    r = AsyncVideoReader(video_path)
    vr_async._live_readers.add(broken)

    with caplog.at_level(logging.ERROR, logger=vr_async.__name__):
        vr_async._shutdown_all_readers()

    assert "Error while shutting down" in caplog.text
    assert r._released, "a failing reader blocked the others"
    assert not list(vr_async._live_readers)


def test_exit_hook_warns_when_teardowns_outlast_the_drain(monkeypatch, caplog):
    """Interpreter exit must be bounded, and say so when it gives up."""
    monkeypatch.setattr(vr_async, "_drain_releases", lambda *a, **k: False)
    with caplog.at_level(logging.WARNING, logger=vr_async.__name__):
        vr_async._shutdown_all_readers()
    assert "Timed out waiting for video reader processes" in caplog.text


def test_drain_is_bounded():
    """A teardown that never finishes must not hang the drain."""
    release = threading.Event()
    vr_async._start_release_thread(release.wait)
    try:
        start = time.perf_counter()
        assert vr_async._drain_releases(timeout=0.2) is False
        assert time.perf_counter() - start < 2.0
    finally:
        release.set()


def test_drain_is_a_noop_when_nothing_is_pending(monkeypatch):
    monkeypatch.setattr(vr_async, "_release_threads", set())
    assert vr_async._drain_releases(timeout=0) is True


def test_exit_hook_is_registered_once(monkeypatch):
    """Constructing more readers must not stack up exit hooks."""
    calls = []
    monkeypatch.setattr(vr_async.atexit, "register", calls.append)
    monkeypatch.setattr(vr_async._register_exit_hook, "_registered", False)

    vr_async._register_exit_hook()
    vr_async._register_exit_hook()

    assert calls == [vr_async._shutdown_all_readers]

"""Process-lifecycle tests for :class:`AsyncVideoReader`.

Every failure mode covered here originally presented as a *hang* rather than an
error: the worker process died, and the caller was left blocked on a future that
nothing remained to resolve. So every wait in this module is bounded — a
regression has to fail the suite, not stall it.
"""

import re
import threading
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest

from asyncvideo import AsyncVideoReader, VideoHandler
from asyncvideo.utils import ReaderError
from asyncvideo.vr_async import mp_ctx

# Long enough for a cold decode on a slow CI runner, short enough that a genuine
# hang ends the run instead of hanging it.
RESULT_TIMEOUT = 30.0
RELEASE_TIMEOUT = 10.0


@pytest.fixture()
def reader(video_path):
    """An ``AsyncVideoReader`` that is always shut down, even if the test fails.

    ``shutdown`` is idempotent, so tests that shut down explicitly are fine too.
    """
    r = AsyncVideoReader(video_path)
    try:
        yield r
    finally:
        r.shutdown()


def _segment_names(reader) -> tuple[str, ...]:
    return tuple(shm.name for shm in reader.shared_mems)


# ---------------------------------------------------------------------------
# Lag injection
#
# Without it, a request usually completes before the next one is submitted, so
# the window in which superseding happens is never open and any assertion about
# cancellation passes trivially.
#
# Where the worker is forked, patching VideoHandler *before* the reader is
# constructed means the child inherits the slowed method. Nothing in ``src`` is
# touched, and the parent is unaffected because only the worker decodes.
#
# This depends on fork copying the parent's memory. Under spawn (Windows) the
# child re-imports the package, so the patch never reaches it and the worker runs
# at full speed. These tests are skipped there rather than left to pass for the
# wrong reason: the parent cancels a superseded future synchronously, so the
# cancellation assertions would still hold with no lag at all, and whether the
# first request was still in flight would come down to timing.
# ---------------------------------------------------------------------------

DECODE_LAG = 0.4

# Deliberately the reader's own context, not multiprocessing's default: macOS
# defaults to spawn while asyncvideo explicitly asks for a fork context, so
# checking the default would skip these tests on a platform where they work.
requires_fork = pytest.mark.skipif(
    mp_ctx.get_start_method() != "fork",
    reason=(
        "lag injection patches the parent process and relies on fork to reach the "
        "worker; under spawn the worker re-imports the module and never sees it"
    ),
)


@pytest.fixture()
def slow_reader_factory(video_path, monkeypatch):
    """Build readers whose worker takes ``DECODE_LAG`` seconds per decode."""
    import asyncvideo._pyav_video_reader as vr_mod

    real_get = vr_mod.VideoHandler.get
    real_getitem = vr_mod.VideoHandler.__getitem__

    def slow_get(self, ts):
        time.sleep(DECODE_LAG)
        return real_get(self, ts)

    def slow_getitem(self, idx):
        time.sleep(DECODE_LAG)
        return real_getitem(self, idx)

    created = []

    def make(**kwargs):
        # patched only for the duration of the fork, so __init__'s own frame-0
        # decode in the parent is slowed too -- harmless, and it keeps the child
        # consistent with what the parent measured
        monkeypatch.setattr(vr_mod.VideoHandler, "get", slow_get)
        monkeypatch.setattr(vr_mod.VideoHandler, "__getitem__", slow_getitem)
        r = AsyncVideoReader(video_path, **kwargs)
        created.append(r)
        return r

    try:
        yield make
    finally:
        for r in created:
            r.shutdown()


@requires_fork
def test_lag_injection_actually_slows_the_worker(slow_reader_factory):
    """Guard the guard: if the patch stopped reaching the child, the tests below
    would silently go back to proving nothing."""
    reader = slow_reader_factory()
    started = time.monotonic()
    reader[10].result(timeout=RESULT_TIMEOUT)
    assert time.monotonic() - started >= DECODE_LAG


@requires_fork
def test_get_supersedes_an_in_flight_index_request(slow_reader_factory, reference):
    """A time request must cancel an index request that is still decoding."""
    packed, height = reference
    reader = slow_reader_factory()
    times = reader.time

    stale = reader[0]
    assert not stale.done(), "decode should still be in flight"
    fresh = reader.get(times[42])

    y, _u, _v = fresh.result(timeout=RESULT_TIMEOUT)
    np.testing.assert_array_equal(y[0], packed[42][:height])
    assert stale.cancelled(), "the superseded request must be cancelled, not served"


@requires_fork
def test_index_supersedes_an_in_flight_get_request(slow_reader_factory, reference):
    """And the reverse: they share one submit path, so either can supersede."""
    packed, height = reference
    reader = slow_reader_factory()
    times = reader.time

    stale = reader.get(times[0])
    assert not stale.done()
    fresh = reader[42]

    y, _u, _v = fresh.result(timeout=RESULT_TIMEOUT)
    np.testing.assert_array_equal(y[0], packed[42][:height])
    assert stale.cancelled()


@requires_fork
def test_only_the_newest_of_many_requests_is_served(slow_reader_factory, reference):
    """Rapid requests, as from dragging a slider: only the last one resolves.

    Also covers the enqueue ordering: ``_submit`` bumps the request id under a
    lock but drains and enqueues outside it, so requests can reach the worker out
    of order. The worker's ``rid < latest_rid`` check is what must discard the
    older ones.
    """
    packed, height = reference
    reader = slow_reader_factory()
    times = reader.time

    futures = []
    for idx in (0, 10, 20, 30):
        futures.append(reader[idx])
    # interleave a time request, so the final winner arrives through get()
    futures.append(reader.get(times[42]))

    y, _u, _v = futures[-1].result(timeout=RESULT_TIMEOUT)
    np.testing.assert_array_equal(y[0], packed[42][:height])
    assert all(f.cancelled() for f in futures[:-1]), (
        "every superseded request must be cancelled"
    )


@requires_fork
def test_requests_for_the_frame_in_flight_share_its_decode(
    slow_reader_factory, reference
):
    """Several consumers reading the same frame must all be served.

    One video displayed in several subplots asks for the same frame once per view.
    Superseding those against each other would leave every view but the last
    without a frame.
    """
    packed, height = reference
    reader = slow_reader_factory()

    # three consumers, all asking for frame 42
    futures = [reader[42], reader[42], reader[42]]
    assert not any(f.done() for f in futures), "decode should still be in flight"

    for future in futures:
        y, _u, _v = future.result(timeout=RESULT_TIMEOUT)
        np.testing.assert_array_equal(y[0], packed[42][:height])


@requires_fork
def test_shared_decode_gives_each_consumer_its_own_frame(
    slow_reader_factory, reference
):
    """One consumer modifying its frame must not change what another was given."""
    packed, height = reference
    reader = slow_reader_factory()

    first, second = reader[42], reader[42]
    y_first = first.result(timeout=RESULT_TIMEOUT)[0]
    y_second = second.result(timeout=RESULT_TIMEOUT)[0]

    assert y_first is not y_second
    y_first[:] = 0

    np.testing.assert_array_equal(y_second[0], packed[42][:height])


@requires_fork
def test_a_new_frame_supersedes_every_shared_request(slow_reader_factory, reference):
    """Sharing applies only to the frame in flight, a new one supersedes them all."""
    packed, height = reference
    reader = slow_reader_factory()

    shared = [reader[0], reader[0]]
    fresh = reader[42]

    y, _u, _v = fresh.result(timeout=RESULT_TIMEOUT)
    np.testing.assert_array_equal(y[0], packed[42][:height])
    assert all(f.cancelled() for f in shared)


@requires_fork
def test_superseded_request_does_not_corrupt_the_result(slow_reader_factory, reference):
    """The winning frame must be intact, not a mix of two decodes.

    The worker writes into one shared buffer, so a superseded decode writing
    while the listener copies out would show up as a frame that matches neither
    request.
    """
    packed, height = reference
    reader = slow_reader_factory()

    for _ in range(5):
        reader[0]
        y, _u, _v = reader[42].result(timeout=RESULT_TIMEOUT)
        np.testing.assert_array_equal(y[0], packed[42][:height])


# ---------------------------------------------------------------------------
# Frame times and time-based access
#
# The worker's handler owns the timestamps. ``get`` resolves a time there, so it
# needs no array on this side; ``time`` is published once by the worker and
# cached here.
# ---------------------------------------------------------------------------


def test_time_matches_the_synchronous_reader(reader, video_path):
    """Parent and worker must agree on what the frame times are."""
    with VideoHandler(video_path) as handler:
        expected = handler.time

    np.testing.assert_array_equal(reader.time, expected)
    assert len(reader.time) == reader.shape[0]


def test_time_is_cached_after_first_read(reader):
    """It arrives once over a queue, so it must be kept rather than re-fetched."""
    first = reader.time
    assert reader.time is first


def test_get_matches_the_synchronous_reader(reader, video_path, reference):
    """get(ts) must resolve to the same frame VideoHandler.get(ts) returns."""
    _, height = reference
    for ts in (0.0, 0.5, 1.5, 3.0):
        y, _u, _v = reader.get(ts).result(timeout=RESULT_TIMEOUT)
        with VideoHandler(video_path) as handler:
            expected = handler.get(ts).to_ndarray()
        np.testing.assert_array_equal(y[0], expected[:height])


def test_get_and_index_agree(reader, reference):
    """Addressing a frame by time or by number must give the same pixels."""
    packed, height = reference
    times = reader.time
    for idx in (0, 7, 42, 99):
        by_index = reader[idx].result(timeout=RESULT_TIMEOUT)[0]
        by_time = reader.get(times[idx]).result(timeout=RESULT_TIMEOUT)[0]
        np.testing.assert_array_equal(by_time, by_index)
        np.testing.assert_array_equal(by_index[0], packed[idx][:height])


def test_get_after_index_request_returns_the_newer_frame(reader, reference):
    """Back-to-back index then time request: the later one wins.

    Deliberately makes no claim about cancellation -- without induced lag the
    first request usually finishes before the second is submitted, so there is no
    supersede window to observe. See the lag-injection tests above for that.
    """
    packed, height = reference
    reader[0]
    y, _u, _v = reader.get(reader.time[42]).result(timeout=RESULT_TIMEOUT)
    np.testing.assert_array_equal(y[0], packed[42][:height])


def test_provided_time_is_used_for_get(video_path, reference):
    """A supplied clock must reach the worker, not be silently dropped."""
    packed, height = reference
    offset = 100.0
    times = np.arange(100) / 30.0 + offset

    r = AsyncVideoReader(video_path, time=times)
    try:
        np.testing.assert_allclose(r.time, times)
        # 100.5 s on this clock is frame 15
        y, _u, _v = r.get(offset + 0.5).result(timeout=RESULT_TIMEOUT)
        np.testing.assert_array_equal(y[0], packed[15][:height])
    finally:
        r.shutdown()


def test_provided_time_wrong_length_raises(video_path):
    """Rejected at construction here: the container declares its frame count."""
    with pytest.raises(ValueError, match="one timestamp per frame"):
        AsyncVideoReader(video_path, time=np.arange(90))


def test_shutdown_without_reading_time_does_not_hang(video_path):
    """The worker publishes the times whether or not anyone collects them.

    A multiprocessing queue holding an unread array keeps its feeder thread
    alive and can block the child at exit, which is why the worker calls
    ``cancel_join_thread``.

    Note this is a smoke test, not a regression test for that call: the fixture
    has 100 frames, so the unread array is under a kilobyte and fits in the pipe
    buffer, and the shutdown stays clean even without ``cancel_join_thread``.
    Reproducing the hang needs an array too large for the buffer -- megabytes,
    i.e. a video with hundreds of thousands of frames.
    """
    started = time.monotonic()
    r = AsyncVideoReader(video_path)
    r[10].result(timeout=RESULT_TIMEOUT)
    r.shutdown()  # never touched r.time
    assert time.monotonic() - started < RELEASE_TIMEOUT


def test_pixel_format_is_not_accepted(video_path):
    """Frames cross in the native format, so the decode format is not a choice."""
    with pytest.raises(TypeError):
        AsyncVideoReader(video_path, pixel_format="rgb24")


# ---------------------------------------------------------------------------
# End-of-stream handling
# ---------------------------------------------------------------------------


def test_sequential_playthrough_then_wrap(reader, reference):
    """Looping playback through the worker process.

    This is the failure seen driving the reader from a render loop: playing to
    the last frame left the worker's decoder at EOF, and the next request failed
    the future with a RuntimeError instead of returning frame 0.
    """
    packed, height = reference
    n_frames = reader.shape[0]

    for idx in range(n_frames):
        y, _u, _v = reader[idx].result(timeout=RESULT_TIMEOUT)
        np.testing.assert_array_equal(y[0], packed[idx][:height])

    # the request that used to raise "reader process failed to decode request N"
    y, _u, _v = reader[0].result(timeout=RESULT_TIMEOUT)
    np.testing.assert_array_equal(y[0], packed[0][:height])


def test_reads_after_playthrough_recover(reader, reference):
    """After a full pass, reads in any direction still resolve."""
    packed, height = reference
    last = reader.shape[0] - 1

    for idx in range(reader.shape[0]):
        reader[idx].result(timeout=RESULT_TIMEOUT)

    for idx in (last, 50, last, 0):
        y, _u, _v = reader[idx].result(timeout=RESULT_TIMEOUT)
        np.testing.assert_array_equal(y[0], packed[idx][:height])


# ---------------------------------------------------------------------------
# Index handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "index",
    [10, (10,), np.int64(10), np.int32(10), slice(10, 11)],
    ids=["int", "tuple", "np_int64", "np_int32", "slice"],
)
def test_index_forms_are_equivalent(reader, reference, index):
    """reader[i], reader[(i,)], numpy ints and slices all select frame 10."""
    packed, height = reference
    y, _u, _v = reader[index].result(timeout=RESULT_TIMEOUT)
    np.testing.assert_array_equal(y[0], packed[10][:height])


def test_empty_tuple_index_raises(reader):
    """An unusable index must raise at the call site, not fail a future."""
    with pytest.raises(IndexError, match="empty tuple"):
        reader[()]


def test_bad_index_does_not_disturb_pending_state(reader, reference):
    """A rejected index must leave the reader usable.

    The index is unpacked before ``__getitem__`` cancels the previous request or
    bumps the request id, so a bad index cannot strand the reader half-updated.
    """
    packed, height = reference
    with pytest.raises(IndexError):
        reader[()]

    y, _u, _v = reader[10].result(timeout=RESULT_TIMEOUT)
    np.testing.assert_array_equal(y[0], packed[10][:height])


def _assert_segment_removed(name: str) -> None:
    """Fail unless the named segment is gone from the system.

    Deliberately only ever *attempts* the attach: on POSIX, successfully
    attaching to a segment also registers it with this process's
    resource_tracker, which would then try to unlink it again at exit.
    """
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name)


def _wait_until(predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


# ---------------------------------------------------------------------------
# Decoding correctness — the reader must return the frame that was asked for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("idx", [0, 5, 42, 99])
def test_int_index_returns_correct_frame(reader, reference, idx):
    """An int index must decode, not hang, and return the *right* frame.

    ``VideoHandler`` returns a bare frame for an int but a list for a slice;
    the worker previously assumed a list and died on the subscript.
    """
    packed, height = reference
    y, _u, _v = reader[(idx,)].result(timeout=RESULT_TIMEOUT)
    np.testing.assert_array_equal(y[0], packed[idx][:height])


@pytest.mark.parametrize("idx", [0, 5, 42, 99])
def test_slice_index_returns_correct_frame(reader, reference, idx):
    packed, height = reference
    y, _u, _v = reader[(slice(idx, idx + 1),)].result(timeout=RESULT_TIMEOUT)
    np.testing.assert_array_equal(y[0], packed[idx][:height])


def test_planar_yuv_planes_are_correctly_subsampled(reader, reference):
    """U and V are half-resolution in both axes for yuv420p."""
    _packed, height = reference
    y, u, v = reader[(10,)].result(timeout=RESULT_TIMEOUT)
    assert y.shape[1:] == (height, y.shape[2])
    assert u.shape[1:] == (y.shape[1] // 2, y.shape[2] // 2)
    assert u.shape == v.shape


def test_packed_yuv_matches_reference(video_path, reference):
    packed, _height = reference
    r = AsyncVideoReader(video_path, yuv_packed=True)
    try:
        frame = r[(10,)].result(timeout=RESULT_TIMEOUT)
        np.testing.assert_array_equal(frame[0], packed[10])
    finally:
        r.shutdown()


# ---------------------------------------------------------------------------
# Worker failure — must surface as an exception, never as a hang
# ---------------------------------------------------------------------------


def test_failed_request_raises_instead_of_hanging(reader):
    """A request the worker cannot serve must fail the future.

    Before the fix the exception killed the worker process outright, so the
    future stayed pending forever and this call never returned.
    """
    # re.escape, not a raw string: ``match`` is a regex, and a raw string only
    # stops Python processing backslashes — it would not stop the regex engine
    # treating a future category name containing ``.`` or ``+`` as a wildcard.
    with pytest.raises(RuntimeError, match=re.escape(ReaderError.type_error.name)):
        reader[("not-an-index",)].result(timeout=RESULT_TIMEOUT)


def test_worker_survives_failed_request(reader, reference):
    """One bad request must not take the reader down with it."""
    packed, height = reference

    with pytest.raises(RuntimeError):
        reader[("not-an-index",)].result(timeout=RESULT_TIMEOUT)

    assert reader._worker.is_alive()

    y, _u, _v = reader[(3,)].result(timeout=RESULT_TIMEOUT)
    np.testing.assert_array_equal(y[0], packed[3][:height])


# ---------------------------------------------------------------------------
# Shutdown — segments released exactly once, whatever the caller does
# ---------------------------------------------------------------------------


def test_shutdown_unlinks_all_segments(reader):
    names = _segment_names(reader)
    assert names  # guard against the fixture silently changing shape

    reader.shutdown()

    for name in names:
        _assert_segment_removed(name)


def test_shutdown_is_idempotent(reader):
    """A second shutdown must not re-unlink and raise FileNotFoundError."""
    names = _segment_names(reader)

    reader.shutdown()
    reader.shutdown()

    for name in names:
        _assert_segment_removed(name)


class _BlockingLock:
    """Real lock that pins the *first* holder inside the critical section.

    Racing N threads and hoping they collide is not a test: they may serialize
    naturally and never overlap, so it passes without ever entering the window
    it claims to cover. This forces the interleaving instead — one thread is
    held inside the critical section while another provably blocks at the door.
    """

    def __init__(self, hold_for: float):
        self._inner = threading.Lock()
        self._hold_for = hold_for
        self.first_holder_inside = threading.Event()
        self._first = True

    def __enter__(self):
        self._inner.acquire()
        if self._first:
            self._first = False
            self.first_holder_inside.set()
            time.sleep(self._hold_for)
        return self

    def __exit__(self, *exc_info):
        self._inner.release()
        return False


def test_concurrent_shutdown_blocks_and_releases_exactly_once(reader):
    """A second shutdown must wait for the first, then not re-unlink.

    ``unlink`` is not idempotent, so a second caller that either slips past the
    guard or acts on a flag set but not yet honoured raises FileNotFoundError.
    """
    names = _segment_names(reader)
    # patch with the lock forcing a race condition
    reader._release_lock = lock = _BlockingLock(hold_for=0.5)
    errors: list[BaseException] = []

    def call_shutdown():
        try:
            reader.shutdown()
        except BaseException as exc:  # noqa: BLE001 - recorded, asserted on below
            errors.append(exc)

    first = threading.Thread(target=call_shutdown)
    first.start()

    # first thread is now inside the critical section, holding the lock
    assert lock.first_holder_inside.wait(RELEASE_TIMEOUT), (
        "first shutdown never started"
    )

    second = threading.Thread(target=call_shutdown)
    second.start()

    # while the first still holds the lock the second must be parked at it, not
    # sailing through into a concurrent teardown
    second.join(timeout=0.1)  # this gives time to reach the lock
    # this makes sure the lock is reached (aka race condition avoided)
    assert second.is_alive(), "second shutdown did not block on the release lock"

    for t in (first, second):
        t.join(timeout=RELEASE_TIMEOUT)
        assert not t.is_alive(), "shutdown deadlocked"

    assert not errors, f"concurrent shutdown raised: {errors}"
    for name in names:
        _assert_segment_removed(name)


def test_shutdown_wait_false_returns_promptly_and_still_releases(reader):
    """``wait=False`` must not block, but must still tear down eventually.

    The teardown cannot run inline: ``_buffer`` is a numpy view onto the shared
    segments and the listener thread may still be copying out of it, so
    unmapping underneath it would be a use-after-free rather than an exception.
    """
    names = _segment_names(reader)

    started = time.monotonic()
    reader.shutdown(wait=False)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"wait=False blocked for {elapsed:.2f}s"

    # the deferred teardown runs on a helper thread; give it a bounded window
    assert _wait_until(lambda: reader._released, RELEASE_TIMEOUT), (
        "deferred teardown never completed"
    )
    for name in names:
        _assert_segment_removed(name)

"""Index of mp4/mov files from their 'moov' box, against the frames ffmpeg decodes.

The files come from generate_edge_case_videos.py, one per timing case; ``nox -s
video_gen`` writes them to tests/test_video/edge_cases/.
"""

import http.server
import threading

import av
import numpy as np
import pytest
from generate_edge_case_videos import CASES, OUTPUT_DIR

from asyncvideo import VideoHandler
from asyncvideo._moov_index import (
    UnsupportedEditListError,
    UnsupportedTiming,
    read_moov_index,
    read_movie_box,
    read_movie_box_http,
)

CASES_BY_OUTCOME = {
    outcome: [name for name, (_, _, expected) in CASES.items() if expected == outcome]
    for outcome in ("moov", "demux", "error")
}


@pytest.fixture(scope="session")
def edge_case_paths():
    paths = {
        name: OUTPUT_DIR / f"{name}.{extension}"
        for name, (_, extension, _) in CASES.items()
    }
    missing = [path.name for path in paths.values() if not path.exists()]
    if missing:
        pytest.skip(f"{missing} missing — run 'nox -s video_gen' to generate test videos")
    return paths


def decode_all(path, stream_index=0):
    """Sorted PTS and frames of everything the decoder outputs: the ground truth."""
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[stream_index]
        for packet in container.demux(stream):
            try:
                frames += [f for f in packet.decode() if f.pts is not None]
            except av.error.InvalidDataError:
                pass  # truncated file: the last packet is cut
    frames.sort(key=lambda f: f.pts)
    pts = np.array([f.pts for f in frames], dtype=np.int64)
    keyframe_pts = [f.pts for f in frames if f.key_frame]
    return pts, [f.to_ndarray() for f in frames], keyframe_pts


@pytest.mark.parametrize("name", CASES_BY_OUTCOME["moov"])
def test_moov_index_matches_decoder(edge_case_paths, name):
    """Indexed from 'moov' at init: no demux thread, PTS equal to the decoded frames'."""
    path = edge_case_paths[name]
    pts, _, keyframe_pts = decode_all(path)
    with VideoHandler(path) as video:
        assert video._index_thread is None
        assert video._keyframe_thread is None
        assert video._index_ready.is_set() and video._pts_keyframe_ready.is_set()
        np.testing.assert_array_equal(video.all_pts, pts)
        assert video.shape[0] == len(pts)
        np.testing.assert_allclose(video.time, pts * float(video.stream.time_base))
        # every keyframe the decoder shows is a sync sample of the index
        assert set(keyframe_pts) <= set(np.asarray(video._keyframe_pts).tolist())


@pytest.mark.parametrize("name", CASES_BY_OUTCOME["moov"])
def test_moov_indexed_frames_are_the_right_ones(edge_case_paths, name):
    """Frames returned by index and by slice are the decoder's, in display order."""
    path = edge_case_paths[name]
    _, frames, _ = decode_all(path)
    n = len(frames)
    with VideoHandler(path) as video:
        for idx in (0, n // 2, n - 1, 1):
            np.testing.assert_array_equal(video[idx].to_ndarray(), frames[idx])
        start = n // 3
        got = video[start : start + 10]
        np.testing.assert_array_equal(
            np.stack([f.to_ndarray() for f in got]), np.stack(frames[start : start + 10])
        )


@pytest.mark.parametrize("name", CASES_BY_OUTCOME["demux"])
def test_unsupported_timing_falls_back_to_demux(edge_case_paths, name):
    path = edge_case_paths[name]
    pts, frames, _ = decode_all(path)
    with VideoHandler(path) as video:
        assert video._index_thread is not None
        video._wait_for_index(15)
        np.testing.assert_array_equal(video.all_pts, pts)
        np.testing.assert_array_equal(video[5].to_ndarray(), frames[5])


@pytest.mark.parametrize("name", CASES_BY_OUTCOME["error"])
def test_unsupported_edit_list_fails_fast(edge_case_paths, name):
    """The error names the file and gives the option that opens it anyway."""
    path = edge_case_paths[name]
    with pytest.raises(UnsupportedEditListError) as excinfo:
        VideoHandler(path)
    message = str(excinfo.value)
    assert path.name in message
    assert "ignore_editlist" in message


@pytest.mark.parametrize("name", CASES_BY_OUTCOME["error"])
def test_ignore_editlist_option_reads_unsupported_edit_list(edge_case_paths, name):
    """The workaround the error suggests: every stored frame, by demuxing."""
    path = edge_case_paths[name]
    with VideoHandler(path, av_open_kwargs={"options": {"ignore_editlist": "1"}}) as video:
        assert video._index_thread is not None
        assert video[0] is not None


def test_stream_index_picks_its_own_track(edge_case_paths):
    """Tracks are matched by track_ID, not position: stream 1 gets the second track."""
    path = edge_case_paths["two_video_tracks"]
    pts, frames, _ = decode_all(path, stream_index=1)
    with VideoHandler(path, stream_index=1) as video:
        assert video._index_thread is None
        # the generator makes stream k (k + 1) times the base size: 320x240 here
        assert video.frame_shape == (240, 320)
        np.testing.assert_array_equal(video.all_pts, pts)


# ---------------------------------------------------------------------------
# Sources: ffmpeg protocol prefixes and HTTP
# ---------------------------------------------------------------------------


def _track_id(path):
    with av.open(str(path)) as container:
        return container.streams.video[0].id


@pytest.mark.parametrize("prefix", ["file:", "async:file:", "cache:file:", "async:cache:file:"])
def test_protocol_prefixes_are_removed(edge_case_paths, prefix):
    path = edge_case_paths["bframes"]
    expected = read_moov_index(str(path), _track_id(path)).pts
    got = read_moov_index(prefix + str(path), _track_id(path)).pts
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("source", ["crypto:/no/such.mp4", "rtsp://host/v.mp4", "/no/such.mp4"])
def test_other_sources_are_unsupported(source):
    with pytest.raises(UnsupportedTiming):
        read_moov_index(source, 1)


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Serves files with HTTP Range support, which http.server lacks."""

    def do_GET(self):  # noqa: N802 -- http.server's naming
        path = self.translate_path(self.path)
        with open(path, "rb") as f:
            data = f.read()
        range_header = self.headers.get("Range")
        if range_header is None or not self.server.supports_range:
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        first, _, last = range_header.removeprefix("bytes=").partition("-")
        first, last = int(first), min(int(last), len(data) - 1)
        body = data[first : last + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {first}-{last}/{len(data)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture(params=[True, False], ids=["range", "no_range"])
def http_server(request, edge_case_paths):
    directory = edge_case_paths["bframes"].parent

    def handler(*args, **kwargs):
        return _RangeHandler(*args, directory=str(directory), **kwargs)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.supports_range = request.param
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def test_moov_over_http(http_server, edge_case_paths):
    """With Range support the header comes over HTTP; without, the reader declines."""
    path = edge_case_paths["bframes"]
    url = f"http://127.0.0.1:{http_server.server_port}/{path.name}"
    if not http_server.supports_range:
        with pytest.raises(UnsupportedTiming, match="Range"):
            read_movie_box_http(url)
        return
    data, moov, file_size = read_movie_box_http(url)
    local_data, local_moov, local_size = read_movie_box(str(path))
    assert data == local_data and file_size == local_size
    assert (moov.payload_start, moov.end) == (local_moov.payload_start, local_moov.end)
    np.testing.assert_array_equal(
        read_moov_index(url, _track_id(path)).pts,
        read_moov_index("async:" + url, _track_id(path)).pts,
    )

"""Generate mp4/mov files with the timing edge cases the 'moov' index must handle.

Each file reproduces one case seen in ffmpeg's FATE samples (B-frames, negative
composition offsets, edit lists that start or end mid-GOP, a stream that starts
on a non-keyframe, a truncated download, fragments, ...) from content made here,
so the tests do not depend on third-party files of unclear licence.

Every case is made with PyAV alone: encode a short clip, then either remux it
(shifting timestamps, dropping packets, muxer flags) or rewrite its Edit List
Box ('elst') in place. The cases and the outcome the reader should have are
listed in ``CASES``.

Usage::

    python tests/generate_edge_case_videos.py

writes the files to ``tests/test_video/edge_cases/``.
"""

from __future__ import annotations

import pathlib
import struct
from collections.abc import Callable

import av
import numpy as np

from asyncvideo._moov_index import children, iter_boxes, parse_timescale

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "test_video" / "edge_cases"

FPS = 30
N_FRAMES = 90
GOP = 30  # keyframe every 30 frames, so a 90-frame clip has 3 GOPs
WIDTH, HEIGHT = 160, 120


# ---------------------------------------------------------------------------
# Encoding and remuxing
# ---------------------------------------------------------------------------


def frame_image(i: int, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """An RGB frame with a bar that moves by 4 pixels per frame.

    The motion makes the encoder use P- and B-frames rather than repeat the
    keyframe.
    """
    image = np.full((height, width, 3), 40, dtype=np.uint8)
    x = (4 * i) % (width - 8)
    image[:, x : x + 8] = (255, 200, 0)
    return image


def encode(
    path: pathlib.Path,
    n_frames: int = N_FRAMES,
    bframes: int = 2,
    container_options: dict | None = None,
    n_video_streams: int = 1,
) -> None:
    """Encode ``n_frames`` with libx264: fixed GOP, ``bframes`` B-frames."""
    x264_params = f"keyint={GOP}:min-keyint={GOP}:scenecut=0:bframes={bframes}"
    with av.open(str(path), "w", options=container_options or {}) as container:
        streams = []
        for k in range(n_video_streams):
            stream = container.add_stream(
                "libx264", rate=FPS, options={"x264-params": x264_params}
            )
            # a different size per stream, so a test can tell them apart
            stream.width, stream.height = WIDTH * (k + 1), HEIGHT * (k + 1)
            stream.pix_fmt = "yuv420p"
            streams.append(stream)
        for i in range(n_frames):
            for k, stream in enumerate(streams):
                image = frame_image(i, WIDTH * (k + 1), HEIGHT * (k + 1))
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                container.mux(stream.encode(frame))
        for stream in streams:
            container.mux(stream.encode())


def remux(
    src: pathlib.Path,
    dst: pathlib.Path,
    keep: Callable[[int, av.Packet], bool] = lambda index, packet: True,
    shift: int = 0,
    container_options: dict | None = None,
) -> None:
    """Copy the packets of ``src`` to ``dst`` without re-encoding.

    Parameters
    ----------
    keep :
        Called with each packet's index in decoding order; False drops it.
    shift :
        Added to every packet's PTS and DTS, in the stream's time base. The
        mp4 muxer turns a stream that does not start at 0 into an edit list.
    """
    with av.open(str(src)) as source, av.open(
        str(dst), "w", options=container_options or {}
    ) as target:
        in_stream = source.streams.video[0]
        out_stream = target.add_stream_from_template(in_stream)
        for index, packet in enumerate(source.demux(in_stream)):
            if packet.dts is None:  # the empty packet that flushes the demuxer
                continue
            if not keep(index, packet):
                continue
            packet.pts += shift
            packet.dts += shift
            packet.stream = out_stream
            target.mux(packet)


def packets(path: pathlib.Path) -> list[tuple[int, int, bool]]:
    """(pts, dts, is_keyframe) of every packet, in decoding order."""
    with av.open(str(path)) as container:
        return [
            (p.pts, p.dts, p.is_keyframe)
            for p in container.demux(container.streams.video[0])
            if p.dts is not None
        ]


# ---------------------------------------------------------------------------
# Rewriting the Edit List Box
# ---------------------------------------------------------------------------


def box_chain(data: bytes, path: list[bytes]) -> list[tuple[int, object]]:
    """(start offset, Box) of each box along ``path``, e.g. moov/trak/edts/elst.

    Takes the first box of each type, so it is meant for single-track files.
    """
    chain = []
    start, end = 0, len(data)
    for box_type in path:
        position = start
        for box in iter_boxes(data, start, end):
            if box.type == box_type:
                chain.append((position, box))
                start, end = box.payload_start, box.end
                break
            position = box.end
        else:
            raise ValueError(f"no {box_type!r} box along {path}")
    return chain


def rewrite_edit_list(
    path: pathlib.Path, edits: list[tuple[float, float | None, int]]
) -> None:
    """Replace the first track's edit list, in place.

    Parameters
    ----------
    edits :
        (segment_duration in seconds, media_time in seconds or None for an
        empty edit, media_rate) per entry.

    The new 'elst' may differ in size from the old one, so the sizes of the
    boxes containing it are updated too. That is only valid when 'moov' comes
    after 'mdat' (PyAV's default): then no sample moves, and the chunk offsets
    stay right.
    """
    data = bytearray(path.read_bytes())
    top = {box.type: position for position, box in _top_level(data)}
    if top[b"moov"] < top[b"mdat"]:
        raise ValueError("'moov' before 'mdat': rewriting it would move the samples")

    moov = box_chain(data, [b"moov"])[0][1]
    trak = box_chain(data, [b"moov", b"trak"])[1][1]
    movie_timescale = parse_timescale(data, children(data, moov)[b"mvhd"])
    media = children(data, children(data, trak)[b"mdia"])
    media_timescale = parse_timescale(data, media[b"mdhd"])

    chain = box_chain(data, [b"moov", b"trak", b"edts", b"elst"])
    elst_start, elst = chain[-1]

    # version 0 entries: uint32 segment_duration, int32 media_time,
    # int16 media_rate_integer, int16 media_rate_fraction (8.6.6.2)
    body = struct.pack(">B3xI", 0, len(edits))
    for duration, media_time, rate in edits:
        body += struct.pack(
            ">Iihh",
            round(duration * movie_timescale),
            -1 if media_time is None else round(media_time * media_timescale),
            rate,
            0,
        )
    new_elst = struct.pack(">I4s", 8 + len(body), b"elst") + body
    delta = len(new_elst) - (elst.end - elst_start)
    data[elst_start : elst.end] = new_elst

    # the boxes that contain 'elst' grow or shrink with it
    for start, _ in chain[:-1]:
        (size,) = struct.unpack_from(">I", data, start)
        struct.pack_into(">I", data, start, size + delta)
    path.write_bytes(bytes(data))


def _top_level(data: bytes):
    position = 0
    for box in iter_boxes(data):
        yield position, box
        position = box.end


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------


def bframes(path):
    """B-frames, so composition order differs from decoding order ('ctts')."""
    encode(path)


def negative_ctts(path):
    """B-frames with negative composition offsets ('ctts' version 1)."""
    encode(path, container_options={"movflags": "negative_cts_offsets"})


def leading_empty_edit(path):
    """The track starts 1 s into the presentation: an empty edit, then the media."""
    source = path.with_suffix(".src.mp4")
    encode(source)
    remux(source, path, shift=_stream_ticks(source, 1.0))
    source.unlink()


def edit_starts_on_bframe(path):
    """Copied from the second keyframe on, but shown from a B-frame after it.

    Like ``ffmpeg -ss <t> -c copy``: the packets start at a keyframe, and the
    edit list hides the frames shown before the requested start.
    """
    source = path.with_suffix(".src.mp4")
    encode(source)
    stream_packets = packets(source)
    second_key = [i for i, (_, _, key) in enumerate(stream_packets) if key][1]
    # the first B-frame after the second keyframe, in decoding order
    first_b = next(
        pts
        for pts, dts, key in stream_packets[second_key + 1 :]
        if pts < stream_packets[second_key + 1][0]
    )
    remux(
        source,
        path,
        keep=lambda index, packet: index >= second_key,
        shift=-first_b,
        container_options={"avoid_negative_ts": "disabled"},
    )
    source.unlink()


def edit_ends_early(path):
    """The edit stops 1.5 frames before the end, between the last frames shown."""
    encode(path)
    rewrite_edit_list(path, [((N_FRAMES - 1.5) / FPS, _first_composition(path), 1)])


def starts_on_non_keyframe(path):
    """The first sample is not a sync sample: the opening keyframe is dropped.

    The decoder outputs nothing until the next keyframe.
    """
    source = path.with_suffix(".src.mp4")
    encode(source)
    remux(source, path, keep=lambda index, packet: index > 0)
    source.unlink()


def truncated(path):
    """'moov' at the front, then cut to 60 %: a partial download.

    The sample tables list frames whose bytes are not in the file.
    """
    encode(path, container_options={"movflags": "faststart"})
    data = path.read_bytes()
    path.write_bytes(data[: int(len(data) * 0.6)])


def two_video_tracks(path):
    """Two video tracks of different sizes, to check stream_index picks the right one."""
    encode(path, n_video_streams=2)


def fragmented(path):
    """Fragmented mp4: the samples live in 'moof' boxes, not in 'moov'."""
    encode(path, container_options={"movflags": "frag_keyframe+empty_moov"})


def two_edits(path):
    """Two edits that map media: 0-1 s, then 2-3 s of the media."""
    encode(path)
    start = _first_composition(path)
    rewrite_edit_list(path, [(1.0, start, 1), (1.0, start + 2.0, 1)])


def dwell_edit(path):
    """A 'dwell' edit (media_rate 0): one frame held for the whole segment."""
    encode(path)
    rewrite_edit_list(path, [(2.0, _first_composition(path) + 1.0, 0)])


def _stream_ticks(path: pathlib.Path, seconds: float) -> int:
    with av.open(str(path)) as container:
        return round(seconds / container.streams.video[0].time_base)


def _first_composition(path: pathlib.Path) -> float:
    """Media time, in seconds, of the first frame shown: the B-frame delay.

    The mp4 muxer's own edit list starts there; an edit list rewritten from
    scratch has to start there too, or it would show nothing at first.
    """
    data = path.read_bytes()
    trak = box_chain(data, [b"moov", b"trak"])[1][1]
    media = children(data, children(data, trak)[b"mdia"])
    media_timescale = parse_timescale(data, media[b"mdhd"])
    try:
        elst = box_chain(data, [b"moov", b"trak", b"edts", b"elst"])[-1][1]
    except ValueError:
        return 0.0  # no edit list: the media starts at 0
    # skip version and flags; the muxer writes version 0 for clips this short
    (entry_count,) = struct.unpack_from(">I", data, elst.payload_start + 4)
    if entry_count == 0:
        return 0.0
    _, media_time = struct.unpack_from(">Ii", data, elst.payload_start + 8)
    return media_time / media_timescale


#: name -> (function, extension, what the reader should do).
#: "moov": indexed from the 'moov' box. "demux": falls back to demuxing.
#: "error": VideoHandler raises UnsupportedEditListError.
CASES = {
    "bframes": (bframes, "mp4", "moov"),
    "bframes_mov": (bframes, "mov", "moov"),
    "negative_ctts": (negative_ctts, "mp4", "moov"),
    "leading_empty_edit": (leading_empty_edit, "mp4", "moov"),
    "edit_starts_on_bframe": (edit_starts_on_bframe, "mp4", "moov"),
    "edit_ends_early": (edit_ends_early, "mp4", "moov"),
    "starts_on_non_keyframe": (starts_on_non_keyframe, "mp4", "moov"),
    "truncated": (truncated, "mp4", "moov"),
    "two_video_tracks": (two_video_tracks, "mp4", "moov"),
    "fragmented": (fragmented, "mp4", "demux"),
    "two_edits": (two_edits, "mp4", "error"),
    "dwell_edit": (dwell_edit, "mp4", "error"),
}


def generate_all(output_dir: pathlib.Path = OUTPUT_DIR) -> dict[str, pathlib.Path]:
    """Write every case to ``output_dir``; return name -> path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (make, extension, _) in CASES.items():
        path = output_dir / f"{name}.{extension}"
        make(path)
        paths[name] = path
        print(f"Saved {path}")
    return paths


if __name__ == "__main__":
    generate_all()

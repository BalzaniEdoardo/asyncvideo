"""Exact frame PTS of an mp4/mov track, taken from the Movie Box ('moov') alone.

Every clause number below refers to ISO/IEC 14496-12:2015, the ISO base media
file format that mp4 is built on. QuickTime (.mov) predates it and uses the same
boxes, which QuickTime calls atoms, for every box this module uses.

How the spec defines a sample's timestamps
------------------------------------------
A track's samples are listed in decoding order, and their times are given by
two tables in the Sample Table Box ('stbl'):

    decoding time      DT(0) = 0,  DT(n+1) = DT(n) + STTS(n)     8.6.1.2
    composition time   CT(n) = DT(n) + CTTS(n)                   8.6.1.3

STTS(n) is the duration of sample n, from the Decoding Time to Sample Box
('stts'). CTTS(n) is its composition offset, from the Composition Time to
Sample Box ('ctts'); when that box is absent, CT = DT (8.6.1.1). Both tables are
run-length encoded as (sample_count, value) pairs.

The composition time is in the media's own timeline. The Edit List Box ('elst',
8.6.6) then maps that media timeline onto the presentation, which is where the
PTS a player shows comes from. The sample tables are complete, so for a file
that is not fragmented the PTS of every frame are known without demuxing: the
'moov' box is a few hundred kB even for hours of video, where a demux goes through (or,
when streaming, downloads) the whole file.

Where the spec leaves the reader a choice, this module does what ffmpeg does,
since the PTS must match the frames PyAV decodes. Each such place says so.
"""

from __future__ import annotations

import os
import struct
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MoovIndex",
    "UnsupportedEditListError",
    "UnsupportedTiming",
    "read_moov_index",
]

# 8.6.1.1, rule 3b: a sample that is decoded but never shown, such as a
# reference-only frame, carries this composition offset (the most negative
# signed 32-bit value).
NON_OUTPUT_SAMPLE_OFFSET = -(2**31)

# 8.6.6.3: an edit whose media_time is -1 is "empty". It maps no media and only
# delays what follows.
EMPTY_EDIT_MEDIA_TIME = -1


class UnsupportedTiming(Exception):
    """The file is valid, but its timing is not fully given by 'moov'.

    The caller should fall back to demuxing the file.
    """


class UnsupportedEditListError(NotImplementedError):
    """The track's edit list is one whose frame timestamps are not handled.

    Supported: no edit list, or an optional leading empty edit followed by one
    normal-rate edit. Several edits that map media (cuts, repeats) and 'dwell'
    edits are not; ffmpeg's handling of them depends on its edit list options
    and on the codec, and a wrong frame index is worse than no reader.
    """


# ---------------------------------------------------------------------------
# Reading binary fields
# ---------------------------------------------------------------------------
#
# Every field in the file is big-endian, most significant byte first (4.2),
# hence the ">" in each struct format and numpy dtype below:
#
#   "I"/">u4"  unsigned int(32)     "i"  signed int(32)
#   "Q"/">u8"  unsigned int(64)     "q"  signed int(64)
#   "h"        signed int(16)       "4s" four raw bytes, a box type

UINT32 = ">I"
UINT64 = ">Q"
BOX_SIZE_AND_TYPE = ">I4s"  # unsigned int(32) size; unsigned int(32) type
EDIT_V0_DURATION_AND_TIME = ">Ii"  # unsigned int(32) segment_duration; int(32) media_time
EDIT_V1_DURATION_AND_TIME = ">Qq"  # unsigned int(64) segment_duration; int(64) media_time
EDIT_MEDIA_RATE = ">hh"  # int(16) media_rate_integer; int(16) media_rate_fraction


# ---------------------------------------------------------------------------
# Box structure, 4.2
# ---------------------------------------------------------------------------
#
#   unsigned int(32) size;       whole box, header included
#   unsigned int(32) type;       four characters, e.g. b"moov"
#   if (size == 1)
#       unsigned int(64) largesize;   the real size, for boxes over 4 GiB
#   else if (size == 0)
#       box extends to the end of the file
#   if (type == 'uuid')
#       unsigned int(8)[16] usertype;


@dataclass
class Box:
    type: bytes
    payload_start: int  # first byte after the header
    end: int  # one past the last byte of the box


def iter_boxes(data: bytes, start: int = 0, end: int | None = None) -> Iterator[Box]:
    """Yield the boxes lying back to back in ``data[start:end]``."""
    end = len(data) if end is None else end
    pos = start
    while pos + 8 <= end:
        size, box_type = struct.unpack_from(BOX_SIZE_AND_TYPE, data, pos)
        header_size = 8
        if size == 1:
            (size,) = struct.unpack_from(UINT64, data, pos + 8)
            header_size = 16
        elif size == 0:
            size = end - pos
        if box_type == b"uuid":
            header_size += 16
        if size < header_size or pos + size > end:
            raise ValueError(f"corrupt {box_type!r} box at byte {pos}")
        yield Box(box_type, pos + header_size, pos + size)
        pos += size


def children(data: bytes, parent: Box) -> dict[bytes, Box]:
    """The child boxes of a container box, by type (first one of each type)."""
    found = {}
    for box in iter_boxes(data, parent.payload_start, parent.end):
        found.setdefault(box.type, box)
    return found


def read_movie_box(path) -> tuple[bytes, Box, int]:
    """Load the bytes of the 'moov' box (8.2.1) of a local file.

    Steps over the top-level boxes before it. Only box headers are loaded on
    the way, so a large Media Data Box ('mdat') placed before 'moov' is
    skipped, not loaded.

    Returns
    -------
    data :
        The payload of the 'moov' box, its header excluded.
    moov :
        The 'moov' box, with offsets into ``data``.
    file_size :
        Size of the whole file in bytes.
    """
    with open(path, "rb") as f:
        file_size = os.fstat(f.fileno()).st_size
        while True:
            pos = f.tell()
            header = f.read(8)
            if len(header) < 8:
                raise ValueError("no 'moov' box in the file")
            size, box_type = struct.unpack(BOX_SIZE_AND_TYPE, header)
            header_size = 8
            if size == 1:
                (size,) = struct.unpack(UINT64, f.read(8))
                header_size = 16
            if box_type == b"moov":
                # size == 0 means "to the end of the file" (4.2)
                payload = f.read() if size == 0 else f.read(size - header_size)
                return payload, Box(b"moov", 0, len(payload)), file_size
            if size == 0:
                raise ValueError("no 'moov' box in the file")
            if size < header_size:
                # a seek by a negative amount would loop over the same bytes
                raise ValueError(f"corrupt {box_type!r} box at byte {pos}")
            f.seek(size - header_size, os.SEEK_CUR)


def _http_range(url: str, start: int, stop: int, timeout: float) -> tuple[bytes, int]:
    """Bytes ``start`` to ``stop - 1`` of ``url``, and the size of the whole file.

    One HTTP Range request. A server that ignores the Range header answers 200
    with the whole file; the response is then closed unread and
    `UnsupportedTiming` raised, so the file is never downloaded here.
    """
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{stop - 1}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 206:
            raise UnsupportedTiming("the server does not support HTTP Range requests")
        # Content-Range: bytes <first>-<last>/<total>, total "*" if unknown
        total = response.headers.get("Content-Range", "").rpartition("/")[2]
        if not total.isdigit():
            raise UnsupportedTiming("the server does not give the file size")
        return response.read(), int(total)


def read_movie_box_http(url: str, timeout: float = 30) -> tuple[bytes, Box, int]:
    """Load the bytes of the 'moov' box (8.2.1) of a file served over HTTP(S).

    Same as `read_movie_box`, with HTTP Range requests in place of seeks: one
    small request per top-level box header, then one for the 'moov' box. The
    Media Data Box ('mdat') is never downloaded. Usually 2 to 4 requests: the
    'moov' box comes right after 'ftyp' in a file written for streaming, and
    after 'mdat' otherwise.

    Raises
    ------
    UnsupportedTiming
        The server does not support Range requests, or does not give the file
        size.
    OSError
        A network or HTTP error (``urllib.error.URLError`` is an OSError).
    """
    pos = 0
    while True:
        # 16 bytes: the size and type, and the largesize that may follow them
        header, file_size = _http_range(url, pos, pos + 16, timeout)
        if len(header) < 8:
            raise ValueError("no 'moov' box in the file")
        size, box_type = struct.unpack_from(BOX_SIZE_AND_TYPE, header)
        header_size = 8
        if size == 1:
            (size,) = struct.unpack_from(UINT64, header, 8)
            header_size = 16
        elif size == 0:
            # size == 0 means "to the end of the file" (4.2)
            size = file_size - pos
        if size < header_size:
            raise ValueError(f"corrupt {box_type!r} box at byte {pos}")
        if box_type == b"moov":
            payload, _ = _http_range(url, pos + header_size, pos + size, timeout)
            return payload, Box(b"moov", 0, len(payload)), file_size
        pos += size
        if pos >= file_size:
            raise ValueError("no 'moov' box in the file")


def full_box_version(data: bytes, box: Box) -> tuple[int, int]:
    """Version of a FullBox (4.2) and the offset where its fields start.

    A FullBox begins with unsigned int(8) version and bit(24) flags.
    """
    return data[box.payload_start], box.payload_start + 4


# ---------------------------------------------------------------------------
# The boxes holding a track's timing
# ---------------------------------------------------------------------------


def parse_timescale(data: bytes, header_box: Box) -> int:
    """'timescale' of a Movie Header Box ('mvhd', 8.2.2) or Media Header Box ('mdhd', 8.4.2).

    Both start the same way, and timescale is the number of time units in one
    second (8.2.2.3, 8.4.2.3):

        version 1: unsigned int(64) creation_time, modification_time;
                   unsigned int(32) timescale;
        version 0: unsigned int(32) creation_time, modification_time;
                   unsigned int(32) timescale;
    """
    version, pos = full_box_version(data, header_box)
    pos += 16 if version == 1 else 8  # skip creation_time, modification_time
    (timescale,) = struct.unpack_from(UINT32, data, pos)
    return timescale


def parse_track_id(data: bytes, track_header_box: Box) -> int:
    """'track_ID' of a Track Header Box ('tkhd', 8.3.2).

        version 1: unsigned int(64) creation_time, modification_time;
                   unsigned int(32) track_ID;
        version 0: unsigned int(32) creation_time, modification_time;
                   unsigned int(32) track_ID;

    ffmpeg uses it as the stream id, which is how a PyAV stream is matched to
    its track.
    """
    version, pos = full_box_version(data, track_header_box)
    pos += 16 if version == 1 else 8
    (track_id,) = struct.unpack_from(UINT32, data, pos)
    return track_id


def parse_run_length_table(data: bytes, table_box: Box, signed: bool) -> NDArray:
    """Expand an 'stts' (8.6.1.2.2) or 'ctts' (8.6.1.3.2) table to one value per sample.

        unsigned int(32) entry_count;
        for each entry:
            unsigned int(32) sample_count;   consecutive samples sharing the value
            unsigned int(32) sample_delta;   stts: duration of each of them
          or  (un)signed int(32) sample_offset;  ctts: CT - DT of each of them
    """
    _, pos = full_box_version(data, table_box)
    (entry_count,) = struct.unpack_from(UINT32, data, pos)
    entries = np.frombuffer(
        data, dtype=">u4", count=2 * entry_count, offset=pos + 4
    ).reshape(entry_count, 2)
    sample_counts = entries[:, 0].astype(np.int64)
    values = entries[:, 1].astype(np.int64)
    if signed:
        # Reinterpret the unsigned 32-bit values as two's complement.
        values = np.where(values >= 2**31, values - 2**32, values)
    return np.repeat(values, sample_counts)


@dataclass
class Edit:
    """One entry of the Edit List Box ('elst'), 8.6.6.3."""

    segment_duration: int  # length on the presentation timeline, movie timescale
    media_time: int  # where in the media it starts, media timescale, composition time
    media_rate: int  # 1: normal playback; 0: 'dwell', a single frame held


def parse_edit_list(data: bytes, edit_list_box: Box) -> list[Edit]:
    """Entries of an Edit List Box ('elst', 8.6.6.2).

        unsigned int(32) entry_count;
        for each entry:
            version 1: unsigned int(64) segment_duration;  int(64) media_time;
            version 0: unsigned int(32) segment_duration;  int(32) media_time;
            int(16) media_rate_integer;
            int(16) media_rate_fraction = 0;
    """
    version, pos = full_box_version(data, edit_list_box)
    (entry_count,) = struct.unpack_from(UINT32, data, pos)
    pos += 4
    duration_and_time = (
        EDIT_V1_DURATION_AND_TIME if version == 1 else EDIT_V0_DURATION_AND_TIME
    )
    entry_size = struct.calcsize(duration_and_time) + 4
    # Not in the spec: some writers set entry_count larger than the entries the
    # box holds. Like ffmpeg, trust the box size over the count.
    entry_count = min(entry_count, (edit_list_box.end - pos) // entry_size)

    edits = []
    for _ in range(entry_count):
        segment_duration, media_time = struct.unpack_from(duration_and_time, data, pos)
        media_rate_integer, _ = struct.unpack_from(
            EDIT_MEDIA_RATE, data, pos + entry_size - 4
        )
        edits.append(Edit(segment_duration, media_time, media_rate_integer))
        pos += entry_size
    return edits


def parse_uint32_list(data: bytes, box: Box) -> NDArray:
    """The table of a FullBox made of an entry_count and that many uint32 values.

    Used for the Sync Sample Box ('stss', 8.6.2.2) and the 32-bit Chunk Offset
    Box ('stco', 8.7.5.2):

        unsigned int(32) entry_count;
        for each entry:
            unsigned int(32) value;   stss: sample_number; stco: chunk_offset
    """
    _, pos = full_box_version(data, box)
    (entry_count,) = struct.unpack_from(UINT32, data, pos)
    return np.frombuffer(data, dtype=">u4", count=entry_count, offset=pos + 4).astype(
        np.int64
    )


def parse_chunk_offsets_64(data: bytes, box: Box) -> NDArray:
    """Chunk offsets of a Chunk Large Offset Box ('co64', 8.7.5.2).

        unsigned int(32) entry_count;
        for each entry:
            unsigned int(64) chunk_offset;
    """
    _, pos = full_box_version(data, box)
    (entry_count,) = struct.unpack_from(UINT32, data, pos)
    return np.frombuffer(data, dtype=">u8", count=entry_count, offset=pos + 4).astype(
        np.int64
    )


def parse_sample_sizes(data: bytes, box: Box) -> NDArray:
    """Size in bytes of every sample, from a Sample Size Box ('stsz', 8.7.3.2).

        unsigned int(32) sample_size;    if not 0, the size of every sample
        unsigned int(32) sample_count;
        if (sample_size == 0)
            for each sample:
                unsigned int(32) entry_size;
    """
    _, pos = full_box_version(data, box)
    sample_size, sample_count = struct.unpack_from(">II", data, pos)
    if sample_size != 0:
        return np.full(sample_count, sample_size, dtype=np.int64)
    return np.frombuffer(
        data, dtype=">u4", count=sample_count, offset=pos + 8
    ).astype(np.int64)


def parse_samples_per_chunk(data: bytes, box: Box, chunk_count: int) -> NDArray:
    """Number of samples in every chunk, from a Sample To Chunk Box ('stsc', 8.7.4.2).

        unsigned int(32) entry_count;
        for each entry:
            unsigned int(32) first_chunk;              1-based (8.7.4.3)
            unsigned int(32) samples_per_chunk;
            unsigned int(32) sample_description_index;

    Each entry starts a run of chunks with the same samples_per_chunk, lasting
    until the next entry's first_chunk (8.7.4.1); the last run lasts until the
    last chunk, whose number comes from the chunk offset box.
    """
    _, pos = full_box_version(data, box)
    (entry_count,) = struct.unpack_from(UINT32, data, pos)
    entries = (
        np.frombuffer(data, dtype=">u4", count=3 * entry_count, offset=pos + 4)
        .reshape(entry_count, 3)
        .astype(np.int64)
    )
    first_chunks = entries[:, 0]
    run_ends = np.append(first_chunks[1:], chunk_count + 1)
    run_lengths = run_ends - first_chunks
    if np.any(run_lengths < 0):
        raise UnsupportedTiming("'stsc' runs are not in increasing chunk order")
    return np.repeat(entries[:, 1], run_lengths)


@dataclass
class VideoTrackTiming:
    """The timing and layout tables of one video track, as stored in the file."""

    movie_timescale: int  # 'mvhd': unit of Edit.segment_duration
    media_timescale: int  # 'mdhd': unit of every other time here
    sample_durations: NDArray  # 'stts', one per sample, decoding order
    composition_offsets: NDArray | None  # 'ctts', one per sample; None if absent
    sample_sizes: NDArray  # 'stsz', bytes, one per sample
    samples_per_chunk: NDArray  # 'stsc', expanded to one per chunk
    chunk_offsets: NDArray  # 'stco' or 'co64', file offset of each chunk
    sync_sample_numbers: NDArray | None  # 'stss', 1-based; None: all are sync
    edits: list[Edit] = field(default_factory=list)  # 'elst'; empty if absent
    is_fragmented: bool = False  # 'mvex' present: more samples live in 'moof' boxes


def read_video_track(data: bytes, moov: Box, track_id: int) -> VideoTrackTiming:
    """Timing tables of the track whose 'tkhd' track_ID is ``track_id``.

    ``data`` and ``moov`` are the 'moov' box as `read_movie_box` or
    `read_movie_box_http` return it; where they came from does not matter here.

    The boxes used, and where the spec places them:

        moov                    Movie Box, 8.2.1
          mvhd                  Movie Header Box, 8.2.2: movie timescale
          mvex                  Movie Extends Box, 8.8.1: the file is fragmented
          trak                  Track Box, 8.3.1, one per track
            tkhd                Track Header Box, 8.3.2: track_ID
            edts / elst         Edit Box 8.6.5 / Edit List Box 8.6.6
            mdia                Media Box, 8.4.1
              mdhd              Media Header Box, 8.4.2: media timescale
              minf / stbl       Media Information 8.4.4 / Sample Table 8.5.1
                stts            Decoding Time to Sample Box, 8.6.1.2
                ctts            Composition Time to Sample Box, 8.6.1.3
                stss            Sync Sample Box, 8.6.2
                stsz            Sample Size Box, 8.7.3
                stsc            Sample To Chunk Box, 8.7.4
                stco / co64     Chunk Offset Box, 8.7.5

    The boxes are looked up at exactly these places.
    """
    movie_children = children(data, moov)
    movie_timescale = parse_timescale(data, movie_children[b"mvhd"])
    is_fragmented = b"mvex" in movie_children

    # 'moov' also holds 'mvhd', 'udta', 'mvex', ...; only 'trak' boxes are tracks
    for track_box in iter_boxes(data, moov.payload_start, moov.end):
        if track_box.type != b"trak":
            continue
        track = children(data, track_box)
        if parse_track_id(data, track[b"tkhd"]) == track_id:
            break
    else:
        raise UnsupportedTiming(f"no track with track_ID {track_id}")

    media = children(data, track[b"mdia"])
    sample_table = children(data, children(data, media[b"minf"])[b"stbl"])

    composition_offsets = None
    if b"ctts" in sample_table:
        # Version 0 offsets are unsigned and version 1 signed (8.6.1.3.2).
        # Treat both as signed, as ffmpeg does: writers do put negative
        # offsets in version 0 boxes.
        composition_offsets = parse_run_length_table(
            data, sample_table[b"ctts"], signed=True
        )
    edits = []
    if b"edts" in track:
        edit_box = children(data, track[b"edts"])
        if b"elst" in edit_box:
            edits = parse_edit_list(data, edit_box[b"elst"])

    # Exactly one of 'stsz' / 'stz2' (8.7.3.1) and one of 'stco' / 'co64'
    # (8.7.5.1) is present.
    if b"stsz" not in sample_table:
        raise UnsupportedTiming("compact sample sizes ('stz2') not handled")
    if b"co64" in sample_table:
        chunk_offsets = parse_chunk_offsets_64(data, sample_table[b"co64"])
    else:
        chunk_offsets = parse_uint32_list(data, sample_table[b"stco"])
    sync_sample_numbers = None
    if b"stss" in sample_table:
        sync_sample_numbers = parse_uint32_list(data, sample_table[b"stss"])

    return VideoTrackTiming(
        movie_timescale=movie_timescale,
        media_timescale=parse_timescale(data, media[b"mdhd"]),
        sample_durations=parse_run_length_table(
            data, sample_table[b"stts"], signed=False
        ),
        composition_offsets=composition_offsets,
        sample_sizes=parse_sample_sizes(data, sample_table[b"stsz"]),
        samples_per_chunk=parse_samples_per_chunk(
            data, sample_table[b"stsc"], len(chunk_offsets)
        ),
        chunk_offsets=chunk_offsets,
        sync_sample_numbers=sync_sample_numbers,
        edits=edits,
        is_fragmented=is_fragmented,
    )


# ---------------------------------------------------------------------------
# From the tables to presentation timestamps
# ---------------------------------------------------------------------------


def movie_to_media_time(duration: int, track: VideoTrackTiming) -> int:
    """Convert a duration from the movie timescale to the media timescale.

    Edit durations are in the movie timescale (8.6.6.3), sample times in the
    media timescale (8.4.2.3); a timescale is units per second, so the result
    is duration * media_timescale / movie_timescale. Rounded to the nearest
    unit (the + movie_timescale // 2), as ffmpeg's av_rescale does.
    """
    numerator = duration * track.media_timescale
    return (numerator + track.movie_timescale // 2) // track.movie_timescale


def sample_end_offsets(track: VideoTrackTiming) -> NDArray:
    """File offset one past the last byte of every sample, in decoding order.

    Samples are stored in chunks (3.1.2); 'stco'/'co64' give where each chunk
    starts in the file (8.7.5.3), 'stsc' how many samples it holds (8.7.4), and
    within a chunk the samples sit back to back, each 'stsz' bytes long (8.7.3).
    So a sample starts at its chunk's offset plus the sizes of the samples
    before it in that chunk.
    """
    sizes = track.sample_sizes
    if track.samples_per_chunk.sum() < len(sizes):
        raise UnsupportedTiming(
            f"'stsc' places {track.samples_per_chunk.sum()} samples but 'stsz' "
            f"lists {len(sizes)}"
        )

    # chunk index of each sample: chunk c repeated samples_per_chunk[c] times
    chunk_index = np.arange(len(track.chunk_offsets))
    chunk_of_sample = np.repeat(chunk_index, track.samples_per_chunk)[: len(sizes)]

    # bytes before each sample, counted from the track's first sample
    bytes_before_in_track = np.cumsum(sizes) - sizes
    # index of each chunk's first sample
    first_sample_of_chunk = np.cumsum(track.samples_per_chunk) - track.samples_per_chunk
    # bytes before each sample, counted from its chunk's first sample
    first_sample_of_my_chunk = first_sample_of_chunk[chunk_of_sample]
    bytes_before_in_chunk = (
        bytes_before_in_track - bytes_before_in_track[first_sample_of_my_chunk]
    )

    sample_start = track.chunk_offsets[chunk_of_sample] + bytes_before_in_chunk
    return sample_start + sizes


@dataclass(frozen=True)
class MoovIndex:
    """Frame timestamps of one track, in the media timescale.

    ffmpeg, and so PyAV, gives an mp4/mov stream the time base
    1 / ``timescale``, so these compare directly to ``frame.pts``.
    """

    #: Sorted PTS of every frame the decoder outputs, one per frame.
    pts: NDArray
    #: Sorted PTS of the sync samples (keyframes), as ffmpeg's packets carry
    #: them: including keyframes that fall outside the edit and so are decoded
    #: but not shown, since a seek can stop at one.
    keyframe_pts: NDArray
    #: Units per second of the times above ('mdhd' timescale).
    timescale: int


def read_moov_index(path: str, track_id: int) -> MoovIndex:
    """Frame and keyframe PTS of an mp4/mov track, without demuxing.

    Parameters
    ----------
    path :
        A local mp4/mov file, or an HTTP(S) URL of one.
    track_id :
        The track's 'tkhd' track_ID; PyAV's ``stream.id`` for mp4/mov.

    Raises
    ------
    UnsupportedTiming
        The timestamps are not fully given by 'moov' (fragmented file,
        inconsistent tables, ...), the source is neither a local file nor an
        HTTP(S) URL, or the server does not support Range requests; demux
        instead.
    UnsupportedEditListError
        The edit list is not one handled here (see that class).
    OSError
        The file could not be opened, or a network or HTTP error.
    """
    # ffmpeg wrapper protocols that pass the inner URL's bytes through
    # unchanged: "async:" fetches ahead in a background thread, "cache:" keeps a
    # local copy. The header is fetched from the inner URL. Wrappers that change
    # the bytes (e.g. "crypto:") are not removed, so they fall back to demuxing.
    while path.startswith(("async:", "cache:")):
        path = path.partition(":")[2]
    if path.startswith(("http://", "https://")):
        data, moov, file_size = read_movie_box_http(path)
    elif os.path.isfile(path):
        data, moov, file_size = read_movie_box(path)
    else:
        raise UnsupportedTiming("not a local file or an HTTP(S) URL")
    track = read_video_track(data, moov, track_id)
    if track.is_fragmented:
        # 8.8.1: the samples in 'moov' are only the first part of the track.
        raise UnsupportedTiming("fragmented: more samples live in 'moof' boxes")

    # 8.6.1.2: DT(0) = 0 and DT(n+1) = DT(n) + STTS(n), where STTS(n) is the
    # duration of sample n.
    sample_durations = track.sample_durations
    n_samples = len(sample_durations)
    if len(track.sample_sizes) != n_samples:
        raise UnsupportedTiming(
            f"'stts' lists {n_samples} samples but 'stsz' lists {len(track.sample_sizes)}"
        )
    decoding_times = np.cumsum(sample_durations) - sample_durations

    # 8.6.1.3: CT(n) = DT(n) + CTTS(n), where CTTS(n) is the composition offset
    # of sample n. Without a 'ctts' box, CT = DT (8.6.1.1).
    composition_offsets = track.composition_offsets
    if composition_offsets is None:
        composition_offsets = np.zeros_like(decoding_times)
    elif len(composition_offsets) != n_samples:
        raise UnsupportedTiming(
            f"'ctts' lists {len(composition_offsets)} samples but 'stts' lists {n_samples}"
        )
    composition_times = decoding_times + composition_offsets

    # Which samples can actually come out of the decoder. Every sample still
    # counts above: the decoding times of later samples depend on the durations
    # of earlier ones, shown or not.
    #
    # 8.6.1.1: non-output samples are never presented.
    decoded = composition_offsets != NON_OUTPUT_SAMPLE_OFFSET
    # A truncated file: the table lists samples whose bytes are not in the file.
    decoded &= sample_end_offsets(track) <= file_size
    # Decoding can start at a sync sample (3.1.12, 8.6.2); without an 'stss' box
    # every sample is one. Samples before the first sync sample lack their
    # reference frames. Not spec: the spec only guarantees decoding from a sync
    # sample onward; ffmpeg's decoders drop the samples before it, so they are
    # never shown.
    is_sync = np.ones(n_samples, dtype=bool)
    if track.sync_sample_numbers is not None:
        if len(track.sync_sample_numbers) == 0:
            raise UnsupportedTiming("no sync sample to start decoding from")
        is_sync[:] = False
        # sample numbers index the sample sequence, starting at 1 (6.2.1)
        sync_index = track.sync_sample_numbers - 1
        is_sync[sync_index[sync_index < n_samples]] = True
        decoded[: sync_index[0]] = False

    # 8.6.6: map the media timeline onto the presentation timeline:
    # PTS = CT - media_time + presentation_start, for the samples inside the
    # edit. Without an 'elst' box, PTS = CT (8.6.5.1).
    edits = list(track.edits)
    presentation_start = 0  # where the media starts on the presentation timeline
    if edits and edits[0].media_time == EMPTY_EDIT_MEDIA_TIME:
        # A leading empty edit delays the whole track by its duration (8.6.6.1).
        presentation_start = movie_to_media_time(edits.pop(0).segment_duration, track)
    if any(edit.media_time == EMPTY_EDIT_MEDIA_TIME for edit in edits):
        raise UnsupportedEditListError(
            "edit list with an empty edit after the first entry"
        )
    if len(edits) > 1:
        raise UnsupportedEditListError(
            f"edit list with {len(edits)} edits that map media; only a single "
            "edit (optionally after an empty one) is supported"
        )
    shown = decoded.copy()
    media_start = 0
    if edits:
        edit = edits[0]
        if edit.media_rate != 1:
            # 8.6.6.3: rate 0 is a 'dwell', one frame held for the whole segment.
            raise UnsupportedEditListError(f"edit with media_rate {edit.media_rate}")
        media_start = edit.media_time
        # Samples composed before media_time or after the segment are not shown.
        # Not spec: edits need not fall on sample times (8.6.6.1 NOTE), so by
        # the spec a frame starting before media_time but lasting into the edit
        # is partly shown; ffmpeg drops it, and so do we.
        shown &= composition_times >= media_start
        # A segment_duration of 0 maps the media from media_time on with no end
        # (8.6.6.1, said of fragmented files, whose length 'moov' cannot know).
        if edit.segment_duration:
            segment_end = media_start + movie_to_media_time(edit.segment_duration, track)
            shown &= composition_times < segment_end

    presentation_times = composition_times - media_start + presentation_start
    # samples are listed in decoding order; sorting gives presentation order
    return MoovIndex(
        pts=np.sort(presentation_times[shown]),
        keyframe_pts=np.sort(presentation_times[decoded & is_sync]),
        timescale=track.media_timescale,
    )

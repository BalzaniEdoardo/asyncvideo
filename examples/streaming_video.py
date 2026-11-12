"""Read frames from a two-hour video without downloading it.

The video is the body camera of a public IBL session: 2.25 GB, about two hours
at 30 fps, served over HTTPS. ``VideoHandler`` opens the URL directly. The frame
index comes from the file's header (its 'moov' box), fetched with a few HTTP
range requests, so the only video data downloaded is the data of the frames
read: here, two seconds from an hour into the session.

The ``async:`` prefix is FFmpeg's read-ahead: it fetches the next bytes in a
background thread while the current ones are decoded. It is optional, but it
makes reading consecutive frames much faster.

Run it::

    python examples/streaming_video.py

``asyncvideo.fetch.DATA_ATTRIBUTION`` records where the data comes from, its
licence, and how to cite it.
"""

from time import perf_counter

import matplotlib.pyplot as plt

from asyncvideo import VideoHandler

URL = (
    "https://ibl.flatironinstitute.org/public/churchlandlab_ucla/Subjects/MFD_09/"
    "2023-10-19/001/raw_video_data/"
    "_iblrig_bodyCamera.raw.e001a1f0-fbf8-49e3-8ecd-6f4c03c26b56.mp4"
)
START, STOP = 3600.0, 3602.0  # seconds into the video: two seconds, an hour in
N_SHOWN = 6  # frames of the window to plot
OUT_PATH = "streaming_video.png"  # written next to wherever you run this

t0 = perf_counter()
with VideoHandler("async:" + URL, pixel_format="rgb24") as video:
    print(f"opened in {perf_counter() - t0:.1f} s: {video.shape} frames x H x W x RGB")
    # No time= here, so times are the video's own timestamps, in seconds.
    print(f"duration {video.time[-1] / 3600:.2f} h")

    window = video.get_slice(START, STOP)
    t0 = perf_counter()
    clip = video[window]  # only now are frames downloaded and decoded
    print(f"read {len(clip)} frames ({START}-{STOP} s) in {perf_counter() - t0:.1f} s")
    times = video.time[window]

# plot frames evenly spaced over the window
step = len(clip) // N_SHOWN
fig, axes = plt.subplots(
    1, N_SHOWN, figsize=(2.6 * N_SHOWN, 2.6), dpi=130, layout="constrained"
)
for ax, frame, t in zip(axes, clip[::step], times[::step]):
    ax.imshow(frame)
    ax.set_title(f"t = {t:.2f} s", fontsize=9, family="monospace")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Streamed from a 2.25 GB video, without downloading it", fontsize=12)
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"wrote {OUT_PATH}")
"""Three cameras, three frame rates, one moment in time.

An IBL session records the mouse from three angles at once, and the cameras do not
share a frame rate:

    left    60 fps    1280x1024
    body    30 fps     640x512
    right  150 fps     640x512

So the same instant is a *different frame number* in each video. Asking by
timestamp removes that arithmetic: each reader is given its camera's own frame
times, and one timestamp then selects the matching frame in every view.

Each reader also owns a decoder process, so the three videos decode at the same
time rather than one after another.

Run it::

    pip install -e ".[docs]"
    python examples/ibl_multiview.py

The clips are downloaded on first run, a few MB in total. The full session is 24 GB
of video; ``asyncvideo.fetch.DATA_ATTRIBUTION`` records where it comes from, its
licence, and how to cite it.
"""

import matplotlib.pyplot as plt
import numpy as np

from asyncvideo import AsyncVideoReader
from asyncvideo.fetch import fetch_times, fetch_video

CAMERAS = ("left", "body", "right")

# The moment to show, in seconds on the experiment clock. The clips are ten seconds
# taken from two minutes into the session, and their timestamps say so -- they run
# from about 120 to 130 s, not from zero. That is what real acquisition timestamps
# look like, and passing them as time= is what lets you ask for 124.5 s directly.
TIMESTAMP = 124.5
OUT_PATH = "ibl_multiview.png"  # written next to wherever you run this

# One reader per camera. time= hands each one its own clock, so get() below means
# the same instant in all three despite the different frame rates.
readers = {
    camera: AsyncVideoReader(fetch_video(camera), time=fetch_times(camera))
    for camera in CAMERAS
}

try:
    # Issue every request before waiting for any of them. That is what lets the
    # three decodes overlap; waiting on each in turn would serialise them.
    futures = {camera: reader.get(TIMESTAMP) for camera, reader in readers.items()}

    # Now collect. result() waits for the frame, to_rgb converts the native YUV,
    # and [0] drops the leading single-frame axis.
    views = {
        camera: readers[camera].to_rgb(future.result(timeout=60))[0]
        for camera, future in futures.items()
    }

    # The same timestamp is a different frame number in each camera -- the
    # bookkeeping you would otherwise be doing by hand. The timestamps are on the
    # experiment clock, so they start where the clip was cut from, not at zero.
    for camera, reader in readers.items():
        frame = int(np.searchsorted(reader.time, TIMESTAMP, side="right") - 1)
        fps = 1.0 / float(np.median(np.diff(reader.time)))
        print(
            f"{camera:<6} {fps:5.1f} fps  starts at t={reader.time[0]:.2f}s "
            f"{reader.time[:3].round(3)}  ->  t={TIMESTAMP} is frame {frame}"
        )
finally:
    # each reader owns a process
    for reader in readers.values():
        reader.shutdown()

# constrained layout, so the panel titles and the suptitle get room instead of
# being clipped
fig, axes = plt.subplots(
    1, len(views), figsize=(4.5 * len(views), 4.2), dpi=130, layout="constrained"
)
for ax, (camera, frame) in zip(axes, views.items()):
    ax.imshow(frame)
    ax.set_title(f"{camera} camera", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(f"One session, three cameras, t = {TIMESTAMP:g} s", fontsize=12)
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"wrote {OUT_PATH}")

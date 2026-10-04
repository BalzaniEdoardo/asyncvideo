"""Play a video in a fastplotlib ndwidget without ever blocking the render loop.

The point of `AsyncVideoReader` is that requesting a frame returns immediately, so
the interesting part here is what ``update`` does *not* do: it never calls
``future.result()`` on a future that is still pending. It checks whether the frame
has arrived, draws it if so, and otherwise returns and lets the next render tick
try again. Blocking on ``result()`` inside a render callback would stall the
window, which is exactly the problem the async reader exists to avoid.

Run it::

    pip install -e ".[docs]"
    python examples/fastplotlib_viewer.py

The clip is downloaded on first run, a few MB. See ``asyncvideo.fetch`` for the
data's licence and citations.
"""

import fastplotlib as fpl

from asyncvideo import AsyncVideoReader
from asyncvideo.fetch import fetch_video

CAMERA_LIST = ["body", "left", "right"]

# fetch video and get the path
videos = {}  # this was undefined, so I created a video dict
for cam in CAMERA_LIST:
    path = fetch_video(cam)
    videos[cam] = AsyncVideoReader(path)

# reference space is seconds, one step per frame
time = videos["body"].time
ranges = {"time": (time[0], time[-1], time[1] - time[0])}

ndw = fpl.NDWidget(
    ranges=ranges,
    shape=(1, 3),  # this was needed because i could not use the syntax ndw[name]
)

for i, (name, video) in enumerate(videos.items()):
    ndw[0, i].add_video(
        video,
        dims=("time", "m", "n"),
        display_dims=("m", "n"),
        colorspace=video.colorspace,
        slider_maps={"time": video.time},
        name="video",
    )
    # neither the pixel values nor a row/col axis are interesting for a video
    ndw[0, i].subplot.tooltip.enabled = False
    ndw[0, i].subplot.axes.visible = False

ndw.show()

if __name__ == "__main__":
    fpl.loop.run()

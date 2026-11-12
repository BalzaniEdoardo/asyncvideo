import pathlib

import nox


@nox.session(name="linters", reuse_venv=True)
def linters(session):
    """Run linters"""
    session.install("-e", ".[dev]")
    session.run("ruff", "check", "src", "--ignore", "D")
    session.run("ruff", "check", "tests", "--ignore", "D")


@nox.session(name="linters-fix", reuse_venv=True)
def linters_fix(session):
    """Run linters and auto-fix issues"""
    session.install("-e", ".[dev]")
    session.run("ruff", "check", "src", "--ignore", "D", "--fix")
    session.run("ruff", "check", "tests", "--ignore", "D", "--fix")


@nox.session(name="video_gen", reuse_venv=True)
def video_gen(session):
    """Generate test videos."""
    session.install("-e", ".[dev]", external=True)
    tests_path = pathlib.Path(__file__).parent.resolve() / "tests"
    video_dir = tests_path / "test_video"
    video_dir.mkdir(exist_ok=True)
    generated_video = [
        f"numbered_video_{codec}.{ext}"
        for codec, ext in [
            ("mpeg4", "mp4"),
            ("libx264", "mp4"),
            ("libx264", "mkv"),
            ("mpeg4", "avi"),
            ("vp9", "webm"),
            ("libx265", "mp4"),
            ("av1", "mp4"),
            ("av1", "mkv"),
            ("av1", "webm"),
            ("mpeg2video", "mpg"),
        ]
    ]
    # unevenly spaced frames, so tests can tell real timestamps apart from a
    # uniform grid derived from the nominal frame rate
    generated_video.append("variable_rate_video_libx264.mp4")
    is_in_dir = all((video_dir / name).exists() for name in generated_video)
    session.log(f"videos found: {is_in_dir}")
    if not is_in_dir:
        session.log("Generating numbered videos...")
        session.run("python", str(tests_path / "generate_numbered_video.py"))

    # mp4/mov timing edge cases for the 'moov' index (test_moov_index.py).
    # Twelve tiny clips, a few seconds in all, so always regenerated: the files
    # then always match the generator.
    session.log("Generating edge case videos...")
    session.run("python", str(tests_path / "generate_edge_case_videos.py"))


@nox.session(name="tests", reuse_venv=True)
def tests(session):
    """Run the test suite."""
    session.install("-e", ".[dev]", external=True)
    session.log("Run Tests...")
    cov_args = list(session.posargs) if session.posargs else []
    session.run(
        "pytest",
        *cov_args,
        env={
            "WGPU_FORCE_OFFSCREEN": "1",
        },
    )

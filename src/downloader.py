"""
Downloads a specific time-range segment of audio from a YouTube URL,
converts it to 22050 Hz mono WAV, and saves it to raw/{clip_id}.wav.

Uses yt-dlp's Python API directly (not subprocess). The key detail here:
yt-dlp does NOT take a simple "start_sec/end_sec" option — partial-range
downloads go through `download_ranges` + `download_range_func`, combined
with `force_keyframes_at_cuts` so the cut actually lands on the requested
timestamps rather than the nearest keyframe. Getting this combination
wrong is a common failure mode (silently downloads the whole video, or
produces an empty file) — see yt-dlp issues #8756 and #9328 for examples
of exactly that going wrong.

Usage:
    from pathlib import Path
    from src.downloader import download_segment

    result = download_segment(
        url="https://www.youtube.com/watch?v=...",
        start_sec=90,
        end_sec=120,
        clip_id="en_001",
        output_dir=Path("raw"),
    )
    if result is None:
        # download failed or range unavailable — already logged, move on
        ...
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, download_range_func

console = Console()

MAX_ATTEMPTS = 3
BASE_BACKOFF_SEC = 2.0  # exponential: 2s, 4s, 8s between attempts


def _build_ydl_opts(output_dir: Path, clip_id: str, start_sec: int, end_sec: int) -> dict:
    """Build the yt-dlp options dict for one segment download."""
    # outtmpl deliberately omits the extension — FFmpegExtractAudio's
    # postprocessor controls the final extension (wav), and letting yt-dlp
    # template it in directly can produce double-extensions like .wav.wav
    # on some yt-dlp versions if not handled carefully.
    out_template = str(output_dir / clip_id)

    return {
        "format": "bestaudio/best",
        "outtmpl": f"{out_template}.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        # This is the actual range-download mechanism — see module docstring.
        "download_ranges": download_range_func([], [(start_sec, end_sec)]),
        "force_keyframes_at_cuts": True,
        # We raise/handle errors ourselves (see retry loop below), so let
        # yt-dlp surface failures as exceptions rather than silently
        # skipping and returning a "successful" empty result.
        "ignoreerrors": False,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "postprocessor_args": {
            # Force the exact sample rate / channel layout the rest of the
            # pipeline expects, at the postprocessing step.
            "ffmpeg": ["-ar", "22050", "-ac", "1"]
        },
    }


def _attempt_download(url: str, ydl_opts: dict[str, Any], clip_id: str) -> bool:
    """Single download attempt. Returns True on success, raises on failure."""
    with YoutubeDL(ydl_opts) as ydl:  # type: ignore
        ydl.download([url])
    return True


def download_segment(
    url: str,
    start_sec: int,
    end_sec: int,
    clip_id: str,
    output_dir: Path,
) -> Optional[Path]:
    """
    Download the [start_sec, end_sec) range of `url`'s audio as a 22050 Hz
    mono WAV file at output_dir/{clip_id}.wav.

    Retries up to MAX_ATTEMPTS times with exponential backoff on network /
    download failures. If the requested time range simply isn't available
    on the source video (e.g. video is shorter than end_sec), logs a clear
    warning and returns None rather than raising — callers processing a
    batch of clips should be able to skip a bad entry without crashing.

    Returns:
        Path to the resulting WAV file on success, or None on failure.
    """
    if end_sec <= start_sec:
        console.log(
            f"[red]✗[/red] [{clip_id}] Invalid range: end_sec ({end_sec}) "
            f"must be greater than start_sec ({start_sec}). Skipping."
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    expected_path = output_dir / f"{clip_id}.wav"
    ydl_opts = _build_ydl_opts(output_dir, clip_id, start_sec, end_sec)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        console.log(
            f"[cyan]↓[/cyan] [{clip_id}] Download attempt {attempt}/{MAX_ATTEMPTS} "
            f"— range {start_sec}s–{end_sec}s from {url}"
        )
        try:
            _attempt_download(url, ydl_opts, clip_id)

            if expected_path.exists() and expected_path.stat().st_size > 0:
                console.log(
                    f"[green]✓[/green] [{clip_id}] Saved to {expected_path} "
                    f"({expected_path.stat().st_size / 1024:.1f} KB)"
                )
                return expected_path
            else:
                # yt-dlp reported success but the file is missing/empty —
                # this is exactly the failure mode seen when download_ranges
                # silently produces nothing (e.g. requested range starts
                # past the video's actual duration).
                console.log(
                    f"[yellow]⚠[/yellow] [{clip_id}] yt-dlp completed but no "
                    f"valid output file found at {expected_path}. The "
                    f"requested range ({start_sec}s–{end_sec}s) may not "
                    f"exist on this video. Not retrying — skipping clip."
                )
                return None

        except DownloadError as e:
            error_text = str(e).lower()
            # Distinguish "range doesn't exist on this video" (not worth
            # retrying — it'll never succeed) from genuine network blips
            # (worth retrying).
            unrecoverable_markers = (
                "video unavailable",
                "private video",
                "this video is not available",
                "sign in to confirm",
                "requested format is not available",
            )
            if any(marker in error_text for marker in unrecoverable_markers):
                console.log(
                    f"[red]✗[/red] [{clip_id}] Unrecoverable error, not "
                    f"retrying: {e}"
                )
                return None

            if attempt < MAX_ATTEMPTS:
                backoff = BASE_BACKOFF_SEC * (2 ** (attempt - 1))
                console.log(
                    f"[yellow]⚠[/yellow] [{clip_id}] Download error on "
                    f"attempt {attempt}: {e}. Retrying in {backoff:.0f}s..."
                )
                time.sleep(backoff)
            else:
                console.log(
                    f"[red]✗[/red] [{clip_id}] Failed after {MAX_ATTEMPTS} "
                    f"attempts: {e}"
                )
                return None

        except Exception as e:
            # Catch-all so one bad clip never crashes a whole batch run.
            if attempt < MAX_ATTEMPTS:
                backoff = BASE_BACKOFF_SEC * (2 ** (attempt - 1))
                console.log(
                    f"[yellow]⚠[/yellow] [{clip_id}] Unexpected error on "
                    f"attempt {attempt}: {e}. Retrying in {backoff:.0f}s..."
                )
                time.sleep(backoff)
            else:
                console.log(
                    f"[red]✗[/red] [{clip_id}] Failed after {MAX_ATTEMPTS} "
                    f"attempts with unexpected error: {e}"
                )
                return None

    return None


if __name__ == "__main__":
    # Quick manual smoke test — see BUILD_GUIDE.md Prompt 3 verify step.
    # Replace with a real, currently-public YouTube URL before running.
    import sys

    if len(sys.argv) < 2:
        console.log(
            "[yellow]Usage:[/yellow] uv run python -m src.downloader "
            "<youtube_url> [start_sec] [end_sec]"
        )
        sys.exit(1)

    test_url = sys.argv[1]
    test_start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    test_end = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    result = download_segment(
        url=test_url,
        start_sec=test_start,
        end_sec=test_end,
        clip_id="smoke_test",
        output_dir=Path("raw"),
    )
    if result:
        console.log(f"[bold green]Smoke test passed:[/bold green] {result}")
    else:
        console.log("[bold red]Smoke test failed — see logs above.[/bold red]")
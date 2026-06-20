from pathlib import Path

from src.downloader import _build_ydl_opts, download_segment


def test_build_ydl_opts_targets_wav_and_audio_format(tmp_path: Path) -> None:
    opts = _build_ydl_opts(tmp_path, "clip_001", 10, 20)

    assert opts["format"] == "bestaudio/best"
    assert opts["outtmpl"].endswith("clip_001.%(ext)s")
    assert opts["postprocessors"][0]["preferredcodec"] == "wav"
    assert opts["postprocessor_args"]["ffmpeg"] == ["-ar", "22050", "-ac", "1"]


def test_download_segment_rejects_invalid_range(tmp_path: Path) -> None:
    result = download_segment(
        url="https://www.youtube.com/watch?v=not-used",
        start_sec=30,
        end_sec=30,
        clip_id="bad_range",
        output_dir=tmp_path,
    )

    assert result is None

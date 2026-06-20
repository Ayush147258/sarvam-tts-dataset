from pathlib import Path

from src.transcriber import transcribe_clip


def test_transcribe_clip_missing_file_returns_failure(tmp_path: Path) -> None:
    result = transcribe_clip(tmp_path / "missing.wav", "en-IN")

    assert result["success"] is False
    assert result["transcript"] == ""
    assert "not found" in result["error"].lower()

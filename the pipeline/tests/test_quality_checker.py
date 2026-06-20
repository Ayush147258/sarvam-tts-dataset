from pathlib import Path

import numpy as np
import soundfile as sf

from src.quality_checker import check_quality


def _write_wav(path: Path, y: np.ndarray, sr: int = 22050) -> None:
    sf.write(str(path), y.astype(np.float32), sr)


def test_quality_checker_rejects_silence(tmp_path: Path) -> None:
    audio_path = tmp_path / "silence.wav"
    _write_wav(audio_path, np.zeros(22050))

    result = check_quality(audio_path, 20, 0.20, 0.1)

    assert result["passed"] is False
    assert "silence_ratio_too_high" in " ".join(result["reject_reasons"])


def test_quality_checker_accepts_clean_tone(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    y = 0.2 * np.sin(2 * np.pi * 440 * t)
    y[: sr // 10] = 0
    _write_wav(audio_path, y)

    result = check_quality(audio_path, 5, 0.25, 0.1)

    assert set(result) == {
        "snr_db",
        "silence_ratio",
        "clipping_pct",
        "quality_score",
        "passed",
        "reject_reasons",
    }
    assert 0.0 <= result["quality_score"] <= 1.0

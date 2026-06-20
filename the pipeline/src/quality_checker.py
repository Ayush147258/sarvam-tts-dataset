"""
Estimates objective audio quality metrics for a single clip: SNR, silence
ratio, clipping percentage, and a composite quality score. Used to decide
whether a segmented clip is clean enough to keep, before it ever reaches
the Sarvam ASR/diarization/emotion-tagging stages — rejecting a noisy or
clipped clip here is cheaper than discovering it after spending API calls
on it.

These are deliberately simple, fast heuristics (percentile-based SNR,
energy-threshold silence, sample-magnitude clipping) — not a full
perceptual audio quality model. They're meant to catch obviously bad
clips automatically and flag borderline ones; they are NOT a substitute
for actually listening (see BUILD_GUIDE.md Phase 6). Treat a "passed"
result as "worth listening to," not as "confirmed good."

Usage:
    from pathlib import Path
    from src.quality_checker import check_quality

    result = check_quality(
        Path("processed/en_001_seg0.wav"),
        snr_threshold_db=20,
        silence_ratio_max=0.20,
        clipping_max_pct=0.1,
    )
    # result: {snr_db, silence_ratio, clipping_pct, quality_score,
    #          passed, reject_reasons}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.config import PROJECT_ROOT

os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".numba_cache"))

try:
    import librosa
except ImportError as e:
    raise ImportError(
        "librosa is required for src/quality_checker.py. Install with: "
        "uv add librosa"
    ) from e

from rich.console import Console

console = Console()

# --- frame-level analysis parameters ---
FRAME_LENGTH = 2048   # samples per analysis frame, ~93ms at 22050Hz
HOP_LENGTH = 512       # ~75% overlap between frames

# --- SNR estimation parameters ---
# Percentile-based noise floor estimate: the bottom NOISE_FLOOR_PERCENTILE
# of frame energies (by RMS) is treated as "noise," and the top
# SIGNAL_PERCENTILE is treated as "signal." This is a coarse approximation
# — it assumes the quietest ~15% of frames are mostly silence/background
# noise and the loudest ~15% are mostly voiced speech, which holds
# reasonably well for single-speaker clips with natural pauses (exactly
# what src/segmenter.py is designed to produce) but would NOT hold for,
# e.g., a clip that is speech-saturated with no pauses at all.
NOISE_FLOOR_PERCENTILE = 15
SIGNAL_PERCENTILE = 85

# --- silence detection parameters ---
# A frame is "silent" if EITHER of two conditions hold:
#   (a) its RMS energy is more than SILENCE_THRESHOLD_DB_BELOW_PEAK below
#       the clip's own peak RMS (catches quiet pauses within an otherwise
#       normal-volume clip), OR
#   (b) its absolute RMS energy is below SILENCE_ABSOLUTE_RMS_FLOOR
#       (catches the case where the ENTIRE clip is uniformly quiet, e.g.
#       a clip that's silence/noise-floor from start to end — here there
#       is no "loud" frame to be relatively quiet against, so the
#       peak-relative check alone would wrongly report 0% silence).
# Verified empirically: a synthetic all-near-silence test file scored
# silence_ratio=0.0 under peak-relative-only detection (bug), and correctly
# scored ~1.0 once the absolute floor was added.
SILENCE_THRESHOLD_DB_BELOW_PEAK = 40
SILENCE_ABSOLUTE_RMS_FLOOR = 0.003  # empirical floor for "no real signal present"

# --- clipping detection parameters ---
# Samples at or above this fraction of full scale (1.0 for float audio)
# are counted as clipped. 0.999 rather than exactly 1.0 to catch
# near-clipping that softclip/limiter processing sometimes leaves just
# under the hard ceiling.
CLIPPING_AMPLITUDE_THRESHOLD = 0.999

# --- composite quality_score weights ---
# Documented here so the weighting is visible and adjustable, not buried.
# SNR is weighted highest since background noise is the hardest thing to
# fix later and the most damaging to TTS training data. Clipping is
# weighted second-highest since it's a hard, audible defect with no good
# fix. Silence ratio is weighted lowest since some silence is normal and
# expected in natural speech (and src/segmenter.py already trims excess
# silence at segment boundaries) — these weights are a starting point,
# expect to revisit them after the manual listening pass per
# BUILD_GUIDE.md Phase 6.
WEIGHT_SNR = 0.5
WEIGHT_CLIPPING = 0.3
WEIGHT_SILENCE = 0.2


def _score_snr(snr_db: float, threshold_db: float) -> float:
    """
    Map SNR to a 0-1 sub-score. At or above 2x the threshold -> 1.0.
    At or below 0 dB -> 0.0. Linear in between, anchored so the
    threshold itself maps to 0.5 (i.e. "just barely passing" scores as
    middling, not as good).
    """
    if snr_db <= 0:
        return 0.0
    if snr_db >= threshold_db * 2:
        return 1.0
    return float(np.clip(snr_db / (threshold_db * 2), 0.0, 1.0))


def _score_clipping(clipping_pct: float, max_pct: float) -> float:
    """Map clipping percentage to a 0-1 sub-score. 0% clipping -> 1.0,
    at-or-above max_pct -> 0.0, linear in between."""
    if clipping_pct <= 0:
        return 1.0
    if clipping_pct >= max_pct:
        return 0.0
    return float(np.clip(1.0 - (clipping_pct / max_pct), 0.0, 1.0))


def _score_silence(silence_ratio: float, max_ratio: float) -> float:
    """Map silence ratio to a 0-1 sub-score. 0% silence -> 1.0,
    at-or-above max_ratio -> 0.0, linear in between."""
    if silence_ratio <= 0:
        return 1.0
    if silence_ratio >= max_ratio:
        return 0.0
    return float(np.clip(1.0 - (silence_ratio / max_ratio), 0.0, 1.0))


def check_quality(
    audio_path: Path,
    snr_threshold_db: float,
    silence_ratio_max: float,
    clipping_max_pct: float,
) -> Dict:
    """
    Compute SNR, silence ratio, clipping percentage, and a composite
    quality_score for a single audio clip, plus a pass/fail decision
    against the given thresholds.

    Never raises on a bad/unreadable file — returns a result dict with
    passed=False and reject_reasons=["unreadable_file"] instead, so a
    batch run can skip one bad clip without crashing.

    Returns:
        {
            "snr_db": float,
            "silence_ratio": float,       # 0.0-1.0
            "clipping_pct": float,        # 0.0-100.0 (percent)
            "quality_score": float,       # 0.0-1.0 composite
            "passed": bool,
            "reject_reasons": list[str],
        }
    """
    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    except Exception as e:
        console.log(f"[red]✗[/red] Could not read {audio_path}: {e}")
        return {
            "snr_db": None,
            "silence_ratio": None,
            "clipping_pct": None,
            "quality_score": 0.0,
            "passed": False,
            "reject_reasons": ["unreadable_file"],
        }

    if y.size == 0:
        console.log(f"[red]✗[/red] {audio_path} loaded but contains zero samples")
        return {
            "snr_db": None,
            "silence_ratio": None,
            "clipping_pct": None,
            "quality_score": 0.0,
            "passed": False,
            "reject_reasons": ["unreadable_file"],
        }

    reject_reasons: List[str] = []

    # --- frame-level RMS energy, used by both SNR and silence checks ---
    frame_rms = librosa.feature.rms(
        y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    # avoid log(0): floor at a tiny epsilon before converting to dB
    frame_rms_safe = np.maximum(frame_rms, 1e-10)

    # --- SNR estimation (percentile-based) ---
    noise_floor_rms = np.percentile(frame_rms_safe, NOISE_FLOOR_PERCENTILE)
    signal_rms = np.percentile(frame_rms_safe, SIGNAL_PERCENTILE)
    # 20*log10 for amplitude-domain ratio (RMS is an amplitude measure, not power)
    snr_db = float(20 * np.log10(signal_rms / noise_floor_rms))

    if snr_db < snr_threshold_db:
        reject_reasons.append(
            f"snr_below_threshold ({snr_db:.1f}dB < {snr_threshold_db}dB)"
        )

    # --- silence ratio ---
    peak_rms = float(np.max(frame_rms_safe))
    frame_db_below_peak = 20 * np.log10(peak_rms / frame_rms_safe)
    silent_by_relative = frame_db_below_peak > SILENCE_THRESHOLD_DB_BELOW_PEAK
    silent_by_absolute = frame_rms_safe < SILENCE_ABSOLUTE_RMS_FLOOR
    silent_frames = silent_by_relative | silent_by_absolute
    silence_ratio = float(np.mean(silent_frames))

    if silence_ratio > silence_ratio_max:
        reject_reasons.append(
            f"silence_ratio_too_high ({silence_ratio:.2%} > {silence_ratio_max:.2%})"
        )

    # --- clipping ---
    clipped_samples = np.sum(np.abs(y) >= CLIPPING_AMPLITUDE_THRESHOLD)
    clipping_pct = float(100.0 * clipped_samples / y.size)

    if clipping_pct > clipping_max_pct:
        reject_reasons.append(
            f"clipping_too_high ({clipping_pct:.3f}% > {clipping_max_pct}%)"
        )

    # --- composite quality_score ---
    # Weighted average of the three normalized sub-scores. See module-level
    # WEIGHT_* constants above for the rationale behind these weights.
    snr_score = _score_snr(snr_db, snr_threshold_db)
    clipping_score = _score_clipping(clipping_pct, clipping_max_pct)
    silence_score = _score_silence(silence_ratio, silence_ratio_max)

    quality_score = float(
        WEIGHT_SNR * snr_score
        + WEIGHT_CLIPPING * clipping_score
        + WEIGHT_SILENCE * silence_score
    )

    passed = len(reject_reasons) == 0

    result = {
        "snr_db": round(snr_db, 2),
        "silence_ratio": round(silence_ratio, 4),
        "clipping_pct": round(clipping_pct, 4),
        "quality_score": round(quality_score, 4),
        "passed": passed,
        "reject_reasons": reject_reasons,
    }

    status_icon = "[green]✓[/green]" if passed else "[red]✗[/red]"
    console.log(
        f"{status_icon} {audio_path.name}: SNR={result['snr_db']}dB, "
        f"silence={result['silence_ratio']:.1%}, "
        f"clipping={result['clipping_pct']:.3f}%, "
        f"score={result['quality_score']}, passed={passed}"
        + (f", reasons={reject_reasons}" if reject_reasons else "")
    )

    return result


if __name__ == "__main__":
    # Quick manual smoke test — see BUILD_GUIDE.md Prompt 7 verify step.
    import sys

    if len(sys.argv) < 2:
        console.log(
            "[yellow]Usage:[/yellow] uv run python -m src.quality_checker "
            "<wav_path> [snr_threshold_db] [silence_ratio_max] [clipping_max_pct]"
        )
        sys.exit(1)

    path = Path(sys.argv[1])
    snr_thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 20
    sil_max = float(sys.argv[3]) if len(sys.argv) > 3 else 0.20
    clip_max = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1

    result = check_quality(path, snr_thresh, sil_max, clip_max)
    console.log(f"[bold]Full result:[/bold] {result}")

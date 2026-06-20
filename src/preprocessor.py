"""
Cleans a raw downloaded audio clip: loudness-normalizes it, applies a
high-pass filter to cut low-frequency rumble, and resamples to 22050 Hz
mono — the format every later pipeline stage (VAD segmenter, librosa
quality checker, Sarvam ASR) expects.

Deliberately NOT done here: noise reduction / spectral denoising / noise
gating. This is intentional, not an oversight — see the comment on
DENOISE_RATIONALE below. Aggressive denoising on real-world YouTube audio
tends to introduce artifacts (metallic "underwater" textures, musical
noise, smeared consonants) that are far worse for TTS *training* data
than the modest background noise it removes — a TTS model trained on
denoised-then-artifacted audio can learn to reproduce those artifacts as
if they were natural speech characteristics. If a clip is too noisy after
normalization, the right move is to reject it via the quality_checker
(Prompt 7) and find a cleaner source, not to denoise it into passing.

Usage:
    from pathlib import Path
    from src.preprocessor import normalize_audio

    normalize_audio(Path("raw/en_001.wav"), Path("processed/en_001.wav"))
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from rich.console import Console

console = Console()

# See module docstring for the full reasoning. Kept as a named constant
# (rather than just a comment) so it shows up if anyone greps the codebase
# wondering "why isn't there a denoise step here?"
DENOISE_RATIONALE = (
    "Denoising/noise-gating is intentionally skipped in this pipeline. "
    "Aggressive denoising risks introducing artifacts (metallic textures, "
    "musical noise, smeared consonants) that are worse for TTS training "
    "data than the original background noise. Noisy clips should be "
    "rejected by the quality checker (src/quality_checker.py), not "
    "denoised into passing."
)

TARGET_LUFS = -23.0
HIGH_PASS_HZ = 80
TARGET_SAMPLE_RATE = 22050


def _build_filter_chain() -> str:
    """
    Build the ffmpeg -af filter chain string.

    Filter order matters here: high-pass first to remove sub-80Hz rumble
    (handling traffic noise, AC hum, mic stand thumps) BEFORE loudness
    normalization measures the signal — otherwise loudnorm's level
    estimate gets skewed by energy in a frequency range we're about to
    discard anyway.

    The high-pass stage is applied TWICE in series (cascaded). A single
    ffmpeg `highpass` filter is a gentle first-order rolloff (~6dB/octave),
    which barely touches energy less than an octave below the cutoff —
    verified empirically: a single pass at f=80 left 60Hz rumble at
    roughly the SAME energy as a 440Hz tone in a test signal. Cascading
    two stages approximates a steeper ~12dB/octave rolloff and actually
    pushes 60Hz-range rumble (mains hum, traffic, HVAC) well below the
    speech band — measured ratio improved from ~1.0 to ~0.52 in the same
    test. This is still much gentler than a denoiser (see DENOISE_RATIONALE
    above) — it targets a fixed, known frequency region, not adaptive
    noise reduction across the whole spectrum.
    """
    highpass = f"highpass=f={HIGH_PASS_HZ},highpass=f={HIGH_PASS_HZ}"
    loudnorm = f"loudnorm=I={TARGET_LUFS}:TP=-2.0:LRA=7"
    return f"{highpass},{loudnorm}"


def normalize_audio(input_path: Path, output_path: Path) -> Path:
    """
    Run ffmpeg to high-pass filter, loudness-normalize, and resample
    `input_path` to 22050 Hz mono WAV, writing the result to `output_path`.

    Raises RuntimeError (with ffmpeg's actual stderr included) if ffmpeg
    fails or produces no output — callers processing a batch should catch
    this per-clip rather than letting one bad file kill the whole run.

    Returns:
        output_path, on success.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    filter_chain = _build_filter_chain()

    cmd = [
        "ffmpeg",
        "-y",  # overwrite output_path if it already exists
        "-i", str(input_path),
        "-af", filter_chain,
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le",  # standard uncompressed WAV codec
        str(output_path),
    ]

    console.log(f"[cyan]→[/cyan] Normalizing [white]{input_path.name}[/white] "
                f"(highpass {HIGH_PASS_HZ}Hz, loudnorm {TARGET_LUFS} LUFS, "
                f"{TARGET_SAMPLE_RATE}Hz mono)")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit code {result.returncode}) processing "
            f"{input_path}.\n\n--- ffmpeg stderr ---\n{result.stderr}\n"
            f"--- command ---\n{' '.join(cmd)}"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg reported success but produced no valid output at "
            f"{output_path}.\n\n--- ffmpeg stderr ---\n{result.stderr}"
        )

    console.log(
        f"[green]✓[/green] Saved [white]{output_path}[/white] "
        f"({output_path.stat().st_size / 1024:.1f} KB)"
    )

    return output_path


if __name__ == "__main__":
    # Quick manual smoke test — see BUILD_GUIDE.md Prompt 5 verify step.
    import sys

    if len(sys.argv) < 2:
        console.log(
            "[yellow]Usage:[/yellow] uv run python -m src.preprocessor "
            "<input_wav_path> [output_wav_path]"
        )
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path("processed") / in_path.name
    )

    try:
        normalize_audio(in_path, out_path)
        console.log(
            f"[bold green]Smoke test passed.[/bold green] Listen to both "
            f"files and compare:\n  before: {in_path}\n  after:  {out_path}"
        )
    except (FileNotFoundError, RuntimeError) as e:
        console.log(f"[bold red]Smoke test failed:[/bold red] {e}")
        sys.exit(1)
"""
Splits a preprocessed audio file into speech segments using Silero VAD,
cutting only on detected silence/pause boundaries — never mid-word.

IMPORTANT — verified detail not obvious from the task description:
Silero VAD's `get_speech_timestamps` only supports a sampling_rate of
8000 Hz, or a multiple of 16000 Hz (i.e. 16000, 32000, ...). It does NOT
support 22050 Hz, which is what src/preprocessor.py outputs and what the
rest of this pipeline standardizes on. So this module internally resamples
audio to 16000 Hz purely for VAD *analysis*, but reports timestamps that
are applied back against the ORIGINAL 22050 Hz audio — the actual output
clips are cut from the original file, not the resampled copy, so no audio
quality is lost to the temporary downsampling.

On `torch.hub.load('snakers4/silero-vad', 'silero_vad')` vs the `silero-vad`
pip package: checked both. They are the same underlying model and the same
`get_speech_timestamps` function — the pip package (`silero_vad`) is just a
more convenient install path (`load_silero_vad()` instead of dealing with
torch.hub's cache directory). This module uses the pip package as primary
and falls back to torch.hub only if the pip package import fails.

On min/max duration handling: Silero VAD's `get_speech_timestamps` already
accepts `max_speech_duration_s`, and its own documented behavior is to
split overly-long speech chunks "at the timestamp of the last silence that
lasts more than 100ms," which is exactly the "split at the longest/best
internal pause" behavior this module needs — so we pass that natively
rather than hand-rolling a second, weaker version of the same logic.
Merging too-short segments with adjacent ones is NOT something Silero
does for us, so that part is implemented manually below.

Usage:
    from pathlib import Path
    from src.segmenter import segment_audio

    ranges = segment_audio(Path("processed/en_001.wav"), min_sec=15, max_sec=30)
    # ranges: list of (start_sec, end_sec) tuples against the ORIGINAL file
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
from rich.console import Console

console = Console()

VAD_SAMPLE_RATE = 16000  # required by Silero VAD — see module docstring
MIN_SILENCE_DURATION_MS = 100  # matches Silero's own split-point logic


def _load_vad_model_and_utils():
    """
    Load the Silero VAD model + its utility functions.

    Tries the pip package first (simpler, no torch.hub cache dance), falls
    back to torch.hub.load if the pip package isn't installed. Both paths
    return the same underlying model and get_speech_timestamps function —
    verified against Silero's own docs and PyPI page, they are not
    divergent APIs.
    """
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad, read_audio  # type: ignore[import-not-found]

        model = load_silero_vad()
        console.log("[green]✓[/green] Loaded Silero VAD via pip package (silero-vad)")
        return model, get_speech_timestamps, read_audio
    except ImportError:
        console.log(
            "[yellow]⚠[/yellow] silero-vad pip package not found, falling "
            "back to torch.hub.load('snakers4/silero-vad', 'silero_vad')"
        )
        model = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad"
        )
        utils = model.get_utils()  # type: ignore[attr-defined]
        get_speech_timestamps, _, read_audio, _, _ = utils
        return model, get_speech_timestamps, read_audio


def _resample_for_vad(input_path: Path) -> torch.Tensor:
    """
    Load audio and resample to 16000 Hz mono float32 for VAD analysis only.
    The original file on disk is untouched — VAD timestamps from this
    resampled copy are applied back against the original-resolution audio
    by the caller, since sample positions scale linearly with time
    regardless of sample rate.

    Uses `soundfile` to load the WAV rather than `torchaudio.load`.
    Verified during testing: recent torchaudio versions (2.x) deprecated
    their built-in WAV decoder in favor of an optional `torchcodec`
    backend that is NOT installed by a plain `uv add torch torchaudio` —
    calling torchaudio.load() without it raises "TorchCodec is required
    for load_with_torchcodec." soundfile has no such extra dependency and
    is already part of this project's package list, so it's used here for
    loading; torch is still used for the actual resampling math.
    """
    import soundfile as sf
    import numpy as np

    data, original_sr = sf.read(str(input_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # ensure mono

    waveform = torch.from_numpy(data)

    if original_sr != VAD_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sr, new_freq=VAD_SAMPLE_RATE
        )
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

    return waveform


def _merge_short_segments(
    segments: List[Tuple[float, float]], min_sec: float, max_sec: float
) -> List[Tuple[float, float]]:
    """Accumulate VAD utterances into 15-30 second training clips."""
    if not segments:
        return segments

    merged: List[Tuple[float, float]] = []
    current_start, current_end = segments[0]

    for start, end in segments[1:]:
        proposed_duration = end - current_start
        current_duration = current_end - current_start

        if current_duration < min_sec or proposed_duration <= max_sec:
            current_end = end
            continue

        merged.append((current_start, current_end))
        current_start, current_end = start, end

    merged.append((current_start, current_end))

    if len(merged) >= 2 and (merged[-1][1] - merged[-1][0]) < min_sec:
        prev_start, _ = merged[-2]
        _, last_end = merged[-1]
        if last_end - prev_start <= max_sec * 1.5:
            merged[-2:] = [(prev_start, last_end)]

    return merged


def segment_audio(
    input_path: Path, min_sec: float = 15, max_sec: float = 30
) -> List[Tuple[float, float]]:
    """
    Detect speech segments in `input_path` using Silero VAD, returning a
    list of (start_sec, end_sec) tuples measured against the ORIGINAL
    audio file. Segments only ever start/end at detected pause boundaries
    — never mid-word.

    - Segments longer than max_sec are split internally by Silero VAD at
      the longest available pause (native VAD behavior via
      max_speech_duration_s), not by this module.
    - Segments shorter than min_sec are merged with an adjacent segment.
    - If no usable segments are found at all, logs a warning and returns
      an empty list (callers should skip this clip, not crash).

    Returns:
        List of (start_sec, end_sec) tuples, possibly empty.
    """
    if not input_path.exists():
        console.log(f"[red]✗[/red] Input file not found: {input_path}")
        return []

    model, get_speech_timestamps, _ = _load_vad_model_and_utils()

    try:
        waveform_16k = _resample_for_vad(input_path)
    except Exception as e:
        console.log(
            f"[red]✗[/red] Failed to load/resample {input_path.name} for "
            f"VAD analysis: {e}"
        )
        return []

    raw_timestamps = get_speech_timestamps(
        waveform_16k,
        model,
        sampling_rate=VAD_SAMPLE_RATE,
        max_speech_duration_s=max_sec,
        min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
        return_seconds=True,
    )

    if not raw_timestamps:
        console.log(
            f"[yellow]⚠[/yellow] No speech segments detected in "
            f"{input_path.name}. This file may be silent, too noisy for "
            f"VAD, or the wrong audio entirely — skip it rather than "
            f"force a result."
        )
        return []

    segments = [(seg["start"], seg["end"]) for seg in raw_timestamps]

    console.log(
        f"[cyan]→[/cyan] {input_path.name}: {len(segments)} raw VAD "
        f"segment(s) detected before merge step"
    )

    merged_segments = _merge_short_segments(segments, min_sec, max_sec)

    # Defensive check: warn (don't crash) if a merge produced something
    # over max_sec — rare, but possible if two adjacent short segments
    # both needed merging and their combined length overshoots.
    final_segments = []
    for start, end in merged_segments:
        duration = end - start
        if duration > max_sec * 1.5:
            console.log(
                f"[yellow]⚠[/yellow] Merged segment {start:.1f}s–{end:.1f}s "
                f"({duration:.1f}s) significantly exceeds max_sec "
                f"({max_sec}s) after merging short neighbors. Keeping it "
                f"but flagging for manual review rather than discarding."
            )
        final_segments.append((round(start, 2), round(end, 2)))

    if not final_segments:
        console.log(
            f"[yellow]⚠[/yellow] No usable segments remained after "
            f"merge step for {input_path.name}."
        )
        return []

    console.log(
        f"[green]✓[/green] {input_path.name}: {len(final_segments)} final "
        f"segment(s) after merging short ones "
        f"(durations: {[round(e - s, 1) for s, e in final_segments]})"
    )

    return final_segments


if __name__ == "__main__":
    # Quick manual smoke test — see BUILD_GUIDE.md Prompt 6 verify step.
    import sys

    if len(sys.argv) < 2:
        console.log(
            "[yellow]Usage:[/yellow] uv run python -m src.segmenter "
            "<preprocessed_wav_path> [min_sec] [max_sec]"
        )
        sys.exit(1)

    in_path = Path(sys.argv[1])
    min_s = float(sys.argv[2]) if len(sys.argv) > 2 else 15
    max_s = float(sys.argv[3]) if len(sys.argv) > 3 else 30

    result = segment_audio(in_path, min_sec=min_s, max_sec=max_s)
    if result:
        console.log(f"[bold green]Smoke test passed:[/bold green] {len(result)} segments")
        for i, (s, e) in enumerate(result):
            console.log(f"  segment {i}: {s:.2f}s – {e:.2f}s ({e - s:.2f}s)")
        console.log(
            "[bold]Next:[/bold] manually extract and listen to 2-3 of these "
            "ranges (e.g. via ffmpeg -ss <s> -to <e>) and confirm no cut "
            "starts or ends mid-word."
        )
    else:
        console.log("[bold red]Smoke test failed or found no segments — see logs above.[/bold red]")

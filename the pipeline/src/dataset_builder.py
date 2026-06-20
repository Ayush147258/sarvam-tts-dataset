"""
src/dataset_builder.py

Builds the final HuggingFace Dataset from the manifest.

Logic:
  1. Load data/manifest.json via src.manifest
  2. Filter to clips where manual_review.verdict == "keep"
  3. For each kept clip:
       - Use manually-corrected transcript/emotion if present
       - Fall back to pipeline output if not, and flag manually_verified=False
  4. Validate every row against a Pydantic model before building
  5. Build a datasets.Dataset with an Audio() feature column
  6. Print a summary table

Run as:
    uv run python -m src.dataset_builder
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from datasets import Audio, Dataset, Features, Value
from pydantic import BaseModel, field_validator, model_validator
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from src.config import PROJECT_ROOT
from src.manifest import all_clips, load_manifest

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Valid tag sets (mirrors emotion_tagger.py — single source of truth here)
# ---------------------------------------------------------------------------

VALID_EMOTION_TAGS = {
    "neutral", "happy", "sad", "excited", "angry", "formal",
    "informational", "storytelling", "conversational", "motivational",
    "whisper", "dramatic", "uncertain",
}

VALID_LANGUAGES = {"en-IN", "hi-IN"}

# ---------------------------------------------------------------------------
# Pydantic schema — one row of the final dataset
# ---------------------------------------------------------------------------

class ClipRow(BaseModel):
    """
    Mirrors Section 2.3 metadata schema exactly.
    Validation happens before the row enters the Dataset so bad data
    is caught with a clear error message, not a cryptic Arrow type error.
    """

    id:                 str
    language:           str
    audio_path:         str     # local path; converted to Audio() feature downstream
    transcript:         str
    duration_sec:       float
    emotion:            str
    style:              str
    speaker_gender:     str     # "male" | "female" | "unknown"
    source_url:         str
    source_title:       str
    snr_db:             float
    sample_rate:        int
    quality_score:      float
    manually_verified:  bool
    created_at:         str     # ISO date string

    @field_validator("language")
    @classmethod
    def language_must_be_valid(cls, v: str) -> str:
        if v not in VALID_LANGUAGES:
            raise ValueError(f"language '{v}' not in {VALID_LANGUAGES}")
        return v

    @field_validator("emotion")
    @classmethod
    def emotion_must_be_valid(cls, v: str) -> str:
        if v not in VALID_EMOTION_TAGS:
            raise ValueError(
                f"emotion '{v}' not in valid tag set. "
                "Check manifest — this clip may need re-tagging."
            )
        return v

    @field_validator("duration_sec")
    @classmethod
    def duration_in_range(cls, v: float) -> float:
        if not (10.0 <= v <= 35.0):
            # Warn but don't hard-fail — manual overrides can push outside
            # the nominal 15–30s window by a small margin
            log.warning(f"duration_sec={v:.1f} is outside the nominal 15–30s window.")
        return v

    @field_validator("snr_db")
    @classmethod
    def snr_is_finite(cls, v: float) -> float:
        import math
        if not math.isfinite(v):
            raise ValueError(f"snr_db must be a finite number, got {v}")
        return round(v, 3)

    @field_validator("quality_score")
    @classmethod
    def quality_score_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"quality_score must be 0–1, got {v}")
        return round(v, 4)

    @field_validator("audio_path")
    @classmethod
    def audio_file_must_exist(cls, v: str) -> str:
        if not Path(v).exists():
            raise ValueError(
                f"audio file not found: {v}. "
                "Check that processed/ directory is populated."
            )
        return v

    @field_validator("transcript")
    @classmethod
    def transcript_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("transcript is empty — clip must have a transcript before inclusion.")
        return v.strip()

    @model_validator(mode="after")
    def speaker_gender_valid(self) -> "ClipRow":
        if self.speaker_gender not in ("male", "female", "unknown"):
            log.warning(
                f"speaker_gender='{self.speaker_gender}' is non-standard. "
                "Using 'unknown'."
            )
            self.speaker_gender = "unknown"
        return self


# ---------------------------------------------------------------------------
# Manifest → ClipRow conversion
# ---------------------------------------------------------------------------

def _resolve_transcript(record: dict) -> tuple[str, bool]:
    """
    Return (transcript, manually_verified).
    Prefer manual correction; fall back to ASR output.
    """
    corrected = record["manual_review"].get("transcript_corrected")
    if corrected and corrected.strip():
        return corrected.strip(), True

    asr = record["stages"]["transcription"].get("transcript")
    if asr and asr.strip():
        log.warning(
            f"[{record['id']}] No manual transcript correction — "
            "using raw ASR output. manually_verified=False."
        )
        return asr.strip(), False

    raise ValueError(f"[{record['id']}] No transcript available (ASR and correction both empty).")


def _resolve_emotion(record: dict) -> tuple[str, str]:
    """
    Return (emotion, style).
    Prefer manual correction; fall back to pipeline tag.
    """
    corrected_emotion = record["manual_review"].get("emotion_corrected")

    pipeline_emotion = record["stages"]["emotion_tagging"].get("emotion") or "uncertain"
    pipeline_style   = record["stages"]["emotion_tagging"].get("style")  or "neutral"

    if corrected_emotion and corrected_emotion.strip():
        # Manual correction only covers emotion — style stays as pipeline output
        # (style is harder to mis-tag and rarely needs hand-correction)
        return corrected_emotion.strip(), pipeline_style

    return pipeline_emotion, pipeline_style


def _resolve_audio_path(record: dict) -> str:
    """
    Find the processed WAV path for this clip.
    Checks stages.segment.processed_path first, then falls back to
    a conventional path derived from clip_id.
    """
    segment_path = record["stages"]["segment"].get("processed_path")
    if segment_path:
        path = Path(segment_path)
        candidates = [path] if path.is_absolute() else [PROJECT_ROOT / path, path]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

    # Conventional fallback: processed/{clip_id}.wav
    conventional = PROJECT_ROOT / "processed" / f"{record['id']}.wav"
    if conventional.exists():
        return str(conventional)

    raise ValueError(
        f"[{record['id']}] Cannot find audio file. "
        f"Tried: {segment_path!r} and {conventional}"
    )


def _record_to_row(record: dict) -> ClipRow:
    """
    Convert a manifest record to a validated ClipRow.
    Raises ValueError with a clear message if any required field is missing.
    """
    clip_id = record["id"]

    transcript, manually_verified = _resolve_transcript(record)
    emotion, style                = _resolve_emotion(record)
    audio_path                    = _resolve_audio_path(record)

    src  = record["source_info"]
    qual = record["stages"]["quality_check"]
    seg  = record["stages"]["segment"]

    # speaker_gender: not yet in the pipeline (would need a gender classifier)
    # Default to "unknown" — document in report as a known gap
    speaker_gender = record.get("speaker_gender", "unknown") or "unknown"

    return ClipRow(
        id                = clip_id,
        language          = src.get("language") or "",
        audio_path        = audio_path,
        transcript        = transcript,
        duration_sec      = float(seg.get("duration_sec") or 0.0),
        emotion           = emotion,
        style             = style,
        speaker_gender    = speaker_gender,
        source_url        = src.get("url") or "",
        source_title      = src.get("source_title") or "",
        snr_db            = float(qual.get("snr_db") or 0.0),
        sample_rate       = 22050,
        quality_score     = float(qual.get("quality_score") or 0.0),
        manually_verified = manually_verified,
        created_at        = str(record.get("created_at", date.today().isoformat())[:10]),
    )


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(manifest_path: Path = PROJECT_ROOT / "data" / "manifest.json") -> Dataset:
    """
    Build and return a HuggingFace Dataset from the manifest.

    Steps:
      1. Load manifest
      2. Filter to verdict == "keep"
      3. Validate each record with Pydantic
      4. Convert to columnar dict
      5. Build Dataset with Audio() feature
    """

    # ---- 1. Load manifest --------------------------------------------------
    load_manifest()
    clips = all_clips()

    if not clips:
        raise RuntimeError(
            "Manifest is empty. Run the pipeline on your source clips first."
        )

    # ---- 2. Filter to "keep" -----------------------------------------------
    kept    = [r for r in clips if r["manual_review"]["verdict"] == "keep"]
    skipped = [r for r in clips if r["manual_review"]["verdict"] != "keep"]

    log.info(
        f"[dataset_builder] {len(clips)} total clips in manifest → "
        f"{len(kept)} kept, {len(skipped)} skipped "
        f"(rejected/needs_fix/unreviewed)."
    )

    if not kept:
        raise RuntimeError(
            "No clips with verdict='keep' found in manifest. "
            "Complete the manual listening pass (run_audit.py) first."
        )

    # ---- 3. Validate with Pydantic -----------------------------------------
    rows: list[ClipRow] = []
    validation_errors: list[tuple[str, str]] = []

    for record in kept:
        try:
            row = _record_to_row(record)
            rows.append(row)
        except Exception as e:
            clip_id = record.get("id", "?")
            log.error(f"[dataset_builder] Skipping {clip_id} — validation error: {e}")
            validation_errors.append((clip_id, str(e)))

    if validation_errors:
        console.print(
            f"\n[bold yellow]⚠ {len(validation_errors)} clip(s) failed validation "
            f"and were excluded:[/bold yellow]"
        )
        for cid, err in validation_errors:
            console.print(f"  [red]{cid}[/red]: {err}")

    if not rows:
        raise RuntimeError(
            "All kept clips failed Pydantic validation. "
            "Check the errors above and fix the manifest or audio files."
        )

    # ---- 4. Convert to columnar dict ---------------------------------------
    # datasets.Dataset is built from {column_name: [values...]}
    # The audio column is a list of file paths — Audio() feature decodes them.

    columns: dict[str, list[Any]] = {
        "id":                 [],
        "language":           [],
        "audio":              [],   # will hold file paths; Audio() resolves them
        "transcript":         [],
        "duration_sec":       [],
        "emotion":            [],
        "style":              [],
        "speaker_gender":     [],
        "source_url":         [],
        "source_title":       [],
        "snr_db":             [],
        "sample_rate":        [],
        "quality_score":      [],
        "manually_verified":  [],
        "created_at":         [],
    }

    for row in rows:
        columns["id"].append(row.id)
        columns["language"].append(row.language)
        columns["audio"].append(row.audio_path)   # path string → decoded by Audio()
        columns["transcript"].append(row.transcript)
        columns["duration_sec"].append(row.duration_sec)
        columns["emotion"].append(row.emotion)
        columns["style"].append(row.style)
        columns["speaker_gender"].append(row.speaker_gender)
        columns["source_url"].append(row.source_url)
        columns["source_title"].append(row.source_title)
        columns["snr_db"].append(row.snr_db)
        columns["sample_rate"].append(row.sample_rate)
        columns["quality_score"].append(row.quality_score)
        columns["manually_verified"].append(row.manually_verified)
        columns["created_at"].append(row.created_at)

    # ---- 5. Build Dataset with typed features ------------------------------
    features = Features({
        "id":                Value("string"),
        "language":          Value("string"),
        "audio":             Audio(sampling_rate=22050),
        "transcript":        Value("string"),
        "duration_sec":      Value("float32"),
        "emotion":           Value("string"),
        "style":             Value("string"),
        "speaker_gender":    Value("string"),
        "source_url":        Value("string"),
        "source_title":      Value("string"),
        "snr_db":            Value("float32"),
        "sample_rate":       Value("int32"),
        "quality_score":     Value("float32"),
        "manually_verified": Value("bool"),
        "created_at":        Value("string"),
    })

    dataset = Dataset.from_dict(columns, features=features)

    log.info(f"[dataset_builder] ✅ Built dataset with {len(dataset)} rows.")

    # ---- 6. Print summary --------------------------------------------------
    _print_summary(dataset, rows, validation_errors)

    return dataset


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(
    dataset: Dataset,
    rows: list[ClipRow],
    validation_errors: list[tuple[str, str]],
) -> None:
    """Print a rich summary table to the console."""

    total_duration_sec  = sum(r.duration_sec for r in rows)
    total_duration_min  = total_duration_sec / 60

    # Language balance
    lang_counts: dict[str, int] = {}
    for r in rows:
        lang_counts[r.language] = lang_counts.get(r.language, 0) + 1

    # Emotion distribution
    emotion_counts: dict[str, int] = {}
    for r in rows:
        emotion_counts[r.emotion] = emotion_counts.get(r.emotion, 0) + 1

    # Manual verification split
    n_verified   = sum(1 for r in rows if r.manually_verified)
    n_unverified = len(rows) - n_verified

    console.print("\n" + "═" * 60)
    console.print("[bold cyan]  DATASET BUILD SUMMARY[/bold cyan]")
    console.print("═" * 60)

    # Top-level stats
    stats_table = Table(show_header=False, box=None, padding=(0, 2))
    stats_table.add_column("Label", style="bold")
    stats_table.add_column("Value")

    stats_table.add_row("Total clips",          str(len(dataset)))
    stats_table.add_row("Total duration",       f"{total_duration_min:.1f} min  ({total_duration_sec:.0f} sec)")
    stats_table.add_row("Manually verified",    f"{n_verified} / {len(rows)}  ({100*n_verified/len(rows):.0f}%)")
    stats_table.add_row("Unverified (ASR only)", str(n_unverified))
    stats_table.add_row("Validation errors",    str(len(validation_errors)))

    console.print(stats_table)

    # Language balance
    console.print("\n[bold]Language balance:[/bold]")
    lang_table = Table("Language", "Clips", "% of total", box=None, padding=(0, 2))
    for lang, count in sorted(lang_counts.items()):
        pct = 100 * count / len(rows)
        lang_table.add_row(lang, str(count), f"{pct:.1f}%")
    console.print(lang_table)

    # Emotion distribution
    console.print("\n[bold]Emotion tag distribution:[/bold]")
    emo_table = Table("Emotion", "Clips", "% of total", box=None, padding=(0, 2))
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(rows)
        emo_table.add_row(emotion, str(count), f"{pct:.1f}%")
    console.print(emo_table)

    # Duration distribution (rough quartiles)
    durations = sorted(r.duration_sec for r in rows)
    n = len(durations)
    console.print("\n[bold]Duration distribution:[/bold]")
    dur_table = Table("Stat", "Value", box=None, padding=(0, 2))
    dur_table.add_row("Min",    f"{durations[0]:.1f}s")
    dur_table.add_row("Median", f"{durations[n // 2]:.1f}s")
    dur_table.add_row("Max",    f"{durations[-1]:.1f}s")
    dur_table.add_row("Mean",   f"{total_duration_sec / n:.1f}s")
    console.print(dur_table)

    console.print("\n" + "═" * 60 + "\n")


# ---------------------------------------------------------------------------
# Manual test — run as: uv run python -m src.dataset_builder
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    try:
        dataset = build_dataset()
    except RuntimeError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    # Quick spot-check
    console.print(f"[green]dataset[0] keys:[/green] {list(dataset[0].keys())}")
    console.print(f"[green]dataset[0]['transcript']:[/green] {dataset[0]['transcript'][:80]}")
    console.print(f"[green]dataset[0]['emotion']:[/green]    {dataset[0]['emotion']}")
    console.print(f"[green]dataset[0]['language']:[/green]   {dataset[0]['language']}")
    console.print(f"[green]dataset[0]['audio'] type:[/green] {type(dataset[0]['audio'])}")
    console.print(
        "\n[bold]Verify:[/bold] dataset[0]['audio'] should be a dict with "
        "'array' (numpy array) and 'sampling_rate' (22050).\n"
        "Listen to the audio and confirm it matches the transcript above."
    )

    # Save a local copy for inspection / re-use in upload_hf.py
    out_path = PROJECT_ROOT / "data" / "dataset_cache"
    dataset.save_to_disk(str(out_path))
    console.print(f"[green]✅ Dataset cached to {out_path} for upload_hf.py[/green]")

"""
run_full_pipeline.py

All-in-one script to execute the remaining pipeline stages on the
already-segmented clips in segments/.

Stages executed in order:
  1. Populate manifest from segments/ directory
  2. Quality check all segments
  3. Transcribe passing segments via Sarvam saaras:v3
  4. Emotion tag all transcribed segments via Sarvam sarvam-105b
  5. Auto-audit: set verdict="keep" for clips passing all checks
  6. Print summary

Usage:
    uv run python run_full_pipeline.py
    uv run python run_full_pipeline.py --stage-from transcribe
    uv run python run_full_pipeline.py --limit 5          # test with 5 clips first
    uv run python run_full_pipeline.py --lang en-IN       # only English clips
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import soundfile as sf
import yaml
from rich.console import Console
from rich.table import Table

from src.config import PROJECT_ROOT, settings
from src.manifest import (
    all_clips,
    load_manifest,
    save_manifest,
    summary,
    update_clip,
)
from src.quality_checker import check_quality
from src.transcriber import transcribe_clip
from src.emotion_tagger import tag_emotion

console = Console()

SEGMENTS_DIR = PROJECT_ROOT / "segments"
SOURCES_EN_PATH = PROJECT_ROOT / "data" / "sources_en.yaml"
SOURCES_HI_PATH = PROJECT_ROOT / "data" / "sources_hi.yaml"

STAGES = ("populate", "quality", "transcribe", "emotion", "audit")

# Rate limiting: pause between API calls to avoid 429s
API_DELAY_SEC = 0.3


def _stage_enabled(stage: str, stage_from: str) -> bool:
    return STAGES.index(stage) >= STAGES.index(stage_from)


def _load_source_info(sources_path: Path) -> dict[str, dict]:
    """Load source metadata from a YAML file, keyed by source_id."""
    if not sources_path.exists():
        return {}
    data = yaml.safe_load(sources_path.read_text(encoding="utf-8")) or {}
    sources = data.get("sources", [])
    result = {}
    prefix = "en" if "en" in sources_path.stem else "hi"
    for i, src in enumerate(sources, 1):
        if "REPLACE_ME" in str(src.get("url", "")):
            continue
        source_id = f"{prefix}_src{i:03d}"
        result[source_id] = {
            "url": src.get("url"),
            "source_title": src.get("source_title"),
            "source_type": src.get("source_type"),
            "language": src.get("language") or f"{prefix}-IN",
            "start_sec": src.get("start_sec"),
            "end_sec": src.get("end_sec"),
            "provisional_emotion": src.get("provisional_emotion"),
            "provisional_style": src.get("provisional_style"),
        }
    return result


def stage_populate(lang_filter: str | None = None, limit: int | None = None) -> None:
    """Scan segments/ directory and populate the manifest with entries."""
    console.rule("[bold cyan]Stage 1: Populate Manifest from Segments[/bold cyan]")

    # Load source metadata for enrichment
    all_sources = {}
    all_sources.update(_load_source_info(SOURCES_EN_PATH))
    all_sources.update(_load_source_info(SOURCES_HI_PATH))

    wavs = sorted(SEGMENTS_DIR.glob("*.wav"))
    if lang_filter:
        prefix = lang_filter.split("-")[0].lower()
        wavs = [w for w in wavs if w.stem.startswith(prefix)]

    if limit:
        wavs = wavs[:limit]

    console.print(f"Found {len(wavs)} segment files to process")

    created = 0
    skipped = 0
    for wav in wavs:
        clip_id = wav.stem  # e.g. en_src001_seg001
        info = sf.info(str(wav))
        duration = info.duration

        # Filter to reasonable TTS durations (5-35s)
        if duration < 3 or duration > 40:
            skipped += 1
            continue

        # Extract source_id from clip_id: en_src001_seg001 -> en_src001
        parts = clip_id.rsplit("_seg", 1)
        source_id = parts[0] if len(parts) == 2 else clip_id
        seg_index = int(parts[1]) if len(parts) == 2 else 0

        # Determine language from prefix
        lang = "hi-IN" if clip_id.startswith("hi_") else "en-IN"

        # Get source info if available
        src_info = all_sources.get(source_id, {})
        if not src_info.get("language"):
            src_info["language"] = lang

        # Relative path for manifest
        rel_path = f"segments/{wav.name}"

        update_clip(
            clip_id,
            source_info=src_info,
            stages={
                "download": {"passed": True, "raw_path": f"raw/{source_id}.wav"},
                "preprocess": {"passed": True, "cleaned_path": f"processed/{source_id}.wav"},
                "segment": {
                    "passed": True,
                    "segment_index": seg_index,
                    "processed_path": rel_path,
                    "duration_sec": round(duration, 3),
                },
            },
        )
        created += 1

    save_manifest()
    console.print(f"[green]✓[/green] Manifest populated: {created} clips added, {skipped} skipped (out of range)")


def stage_quality(limit: int | None = None) -> None:
    """Run quality check on all segments that haven't been checked yet."""
    console.rule("[bold cyan]Stage 2: Quality Check[/bold cyan]")

    clips = all_clips()
    to_check = [
        c for c in clips
        if c["stages"]["segment"].get("passed")
        and c["stages"]["quality_check"].get("passed") is None
    ]
    if limit:
        to_check = to_check[:limit]

    console.print(f"Quality checking {len(to_check)} clips...")

    passed = 0
    failed = 0
    for i, clip in enumerate(to_check, 1):
        clip_id = clip["id"]
        seg_path_str = clip["stages"]["segment"].get("processed_path", "")
        seg_path = Path(seg_path_str)
        if not seg_path.is_absolute():
            seg_path = PROJECT_ROOT / seg_path

        if not seg_path.exists():
            console.print(f"[red]✗[/red] {clip_id}: file not found at {seg_path}")
            update_clip(clip_id, stages={"quality_check": {
                "passed": False, "reject_reasons": ["file_not_found"],
                "snr_db": None, "silence_ratio": None, "clipping_pct": None,
                "quality_score": 0.0
            }})
            failed += 1
            continue

        result = check_quality(
            seg_path,
            settings.snr_threshold_db,
            settings.silence_ratio_max,
            settings.clipping_max_pct,
        )
        update_clip(clip_id, stages={"quality_check": result})

        if result["passed"]:
            passed += 1
        else:
            failed += 1

        if i % 50 == 0:
            save_manifest()
            console.print(f"  Progress: {i}/{len(to_check)}")

    save_manifest()
    console.print(f"[green]✓[/green] Quality check done: {passed} passed, {failed} failed")


def stage_transcribe(limit: int | None = None) -> None:
    """Transcribe all quality-passing segments that haven't been transcribed yet."""
    console.rule("[bold cyan]Stage 3: Transcription (Sarvam saaras:v3)[/bold cyan]")

    clips = all_clips()
    to_transcribe = [
        c for c in clips
        if c["stages"]["quality_check"].get("passed") is True
        and c["stages"]["transcription"].get("passed") is None
    ]
    if limit:
        to_transcribe = to_transcribe[:limit]

    console.print(f"Transcribing {len(to_transcribe)} clips...")

    succeeded = 0
    failed = 0
    for i, clip in enumerate(to_transcribe, 1):
        clip_id = clip["id"]
        seg_path_str = clip["stages"]["segment"].get("processed_path", "")
        seg_path = Path(seg_path_str)
        if not seg_path.is_absolute():
            seg_path = PROJECT_ROOT / seg_path

        if not seg_path.exists():
            console.print(f"[red]✗[/red] {clip_id}: file not found")
            update_clip(clip_id, stages={"transcription": {
                "passed": False, "reason": "file_not_found", "transcript": None,
                "model": None, "language_code": None
            }})
            failed += 1
            continue

        lang = clip["source_info"].get("language") or (
            "hi-IN" if clip_id.startswith("hi_") else "en-IN"
        )

        # Check duration — skip if too long for sync API
        duration = clip["stages"]["segment"].get("duration_sec", 0)
        if duration > 35:
            console.print(f"[yellow]⚠[/yellow] {clip_id}: {duration:.1f}s too long for sync API, skipping")
            update_clip(clip_id, stages={"transcription": {
                "passed": False, "reason": f"too_long_{duration:.0f}s",
                "transcript": None, "model": None, "language_code": None
            }})
            failed += 1
            continue

        result = transcribe_clip(seg_path, language_code=lang)

        update_clip(
            clip_id,
            stages={
                "transcription": {
                    "passed": result["success"],
                    "reason": result["error"],
                    "transcript": result["transcript"],
                    "model": "saaras:v3",
                    "language_code": result["language_code"],
                }
            },
        )

        if result["success"]:
            succeeded += 1
        else:
            failed += 1

        # Save every 10 clips and rate limit
        if i % 10 == 0:
            save_manifest()
            console.print(f"  Progress: {i}/{len(to_transcribe)} ({succeeded} OK, {failed} failed)")

        time.sleep(API_DELAY_SEC)

    save_manifest()
    console.print(f"[green]✓[/green] Transcription done: {succeeded} succeeded, {failed} failed")


def stage_emotion(limit: int | None = None) -> None:
    """Tag emotion for all transcribed segments."""
    console.rule("[bold cyan]Stage 4: Emotion Tagging (Sarvam LLM)[/bold cyan]")

    clips = all_clips()
    to_tag = [
        c for c in clips
        if c["stages"]["transcription"].get("passed") is True
        and c["stages"]["transcription"].get("transcript")
        and c["stages"]["emotion_tagging"].get("passed") is None
    ]
    if limit:
        to_tag = to_tag[:limit]

    console.print(f"Tagging emotion for {len(to_tag)} clips...")

    succeeded = 0
    failed = 0
    for i, clip in enumerate(to_tag, 1):
        clip_id = clip["id"]
        transcript = clip["stages"]["transcription"]["transcript"]
        lang = clip["source_info"].get("language") or "en-IN"
        source_type = clip["source_info"].get("source_type") or "unknown"

        result = tag_emotion(
            transcript=transcript,
            language=lang,
            source_type=source_type,
        )

        update_clip(
            clip_id,
            stages={
                "emotion_tagging": {
                    "passed": result["error"] is None,
                    "reason": result["error"],
                    "emotion": result["emotion"],
                    "style": result["style"],
                    "confidence": result["confidence"],
                    "needs_manual_review": result["needs_manual_review"],
                    "model_used": result["model_used"],
                }
            },
        )

        if result["error"] is None:
            succeeded += 1
        else:
            failed += 1

        if i % 10 == 0:
            save_manifest()
            console.print(f"  Progress: {i}/{len(to_tag)} ({succeeded} OK, {failed} failed)")

        time.sleep(API_DELAY_SEC)

    save_manifest()
    console.print(f"[green]✓[/green] Emotion tagging done: {succeeded} succeeded, {failed} failed")


def stage_auto_audit() -> None:
    """Auto-set verdict='keep' for clips that pass all pipeline stages."""
    console.rule("[bold cyan]Stage 5: Auto-Audit[/bold cyan]")

    clips = all_clips()
    kept = 0
    rejected = 0
    for clip in clips:
        clip_id = clip["id"]

        # Already has a human verdict — don't overwrite
        if clip["manual_review"]["verdict"] is not None:
            continue

        # Check all required stages passed
        quality_ok = clip["stages"]["quality_check"].get("passed") is True
        transcription_ok = clip["stages"]["transcription"].get("passed") is True
        has_transcript = bool(clip["stages"]["transcription"].get("transcript"))
        emotion_ok = clip["stages"]["emotion_tagging"].get("passed") is True

        if quality_ok and transcription_ok and has_transcript and emotion_ok:
            update_clip(clip_id, manual_review={"verdict": "keep", "listened": False})
            kept += 1
        elif quality_ok and transcription_ok and has_transcript:
            # Emotion tagging might have failed but transcript is fine — still keep
            update_clip(clip_id, manual_review={"verdict": "keep", "listened": False})
            kept += 1
        else:
            update_clip(clip_id, manual_review={"verdict": "reject"})
            rejected += 1

    save_manifest()
    console.print(f"[green]✓[/green] Auto-audit done: {kept} kept, {rejected} rejected")


def print_final_summary() -> None:
    """Print a summary of the manifest state."""
    console.rule("[bold cyan]Pipeline Summary[/bold cyan]")

    stats = summary()

    table = Table(title="Pipeline Summary", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total clips in manifest", str(stats["total_clips"]))
    for v, c in stats["verdicts"].items():
        table.add_row(f"  Verdict: {v}", str(c))
    for lang, c in stats["languages"].items():
        table.add_row(f"  Language: {lang}", str(c))
    for stage, c in stats["stages_passed"].items():
        table.add_row(f"  {stage} passed", str(c))

    console.print(table)

    # Compute total duration of 'keep' clips
    clips = all_clips()
    keep_clips = [c for c in clips if c["manual_review"]["verdict"] == "keep"]
    en_dur = sum(c["stages"]["segment"].get("duration_sec", 0) for c in keep_clips if c["id"].startswith("en_"))
    hi_dur = sum(c["stages"]["segment"].get("duration_sec", 0) for c in keep_clips if c["id"].startswith("hi_"))

    console.print(f"\n[bold]Duration of 'keep' clips:[/bold]")
    console.print(f"  English: {en_dur/60:.1f} min ({en_dur:.0f}s)")
    console.print(f"  Hindi:   {hi_dur/60:.1f} min ({hi_dur:.0f}s)")
    console.print(f"  Total:   {(en_dur+hi_dur)/60:.1f} min ({en_dur+hi_dur:.0f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run remaining pipeline stages on segmented clips.")
    parser.add_argument("--stage-from", choices=STAGES, default="populate",
                        help="Resume from this stage")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N clips per stage (for testing)")
    parser.add_argument("--lang", default=None,
                        help="Only process clips for this language (e.g. en-IN or hi-IN)")
    args = parser.parse_args()

    load_manifest()

    if _stage_enabled("populate", args.stage_from):
        stage_populate(lang_filter=args.lang, limit=args.limit)
        # Reload after populating
        load_manifest()

    if _stage_enabled("quality", args.stage_from):
        stage_quality(limit=args.limit)
        load_manifest()

    if _stage_enabled("transcribe", args.stage_from):
        stage_transcribe(limit=args.limit)
        load_manifest()

    if _stage_enabled("emotion", args.stage_from):
        stage_emotion(limit=args.limit)
        load_manifest()

    if _stage_enabled("audit", args.stage_from):
        stage_auto_audit()

    print_final_summary()
    console.print("\n[bold green]Pipeline complete.[/bold green]")


if __name__ == "__main__":
    main()

"""
src/manifest.py

Maintains data/manifest.json — the single source of truth for every clip
that has passed through any stage of the pipeline.

Design principles:
  - Append-only-ish: update_clip() merges new fields into an existing entry
    rather than overwriting the whole record. Old fields are never silently
    deleted — you must explicitly pass None to clear a field.
  - The manual_review sub-object is intentionally left empty by the pipeline.
    It is filled in by hand during the listening pass (scripts/run_audit.py).
  - Every write goes through save_manifest() which does an atomic
    write-to-temp-then-rename so a crash mid-write never corrupts the file.
  - Thread safety: a threading.Lock guards in-memory state so the pipeline
    can call update_clip() from a ThreadPoolExecutor without races.
    (For multiprocess parallelism you'd need file-level locking — out of
    scope for this project but noted here.)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

from src.config import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"

# ---------------------------------------------------------------------------
# Schema: empty record templates
# ---------------------------------------------------------------------------

def _empty_source_info() -> dict:
    return {
        "url":                    None,   # str  — YouTube URL
        "source_title":           None,   # str  — video title
        "source_type":            None,   # str  — e.g. "tedx_talk"
        "language":               None,   # str  — "en-IN" | "hi-IN"
        "start_sec":              None,   # int  — segment start in source video
        "end_sec":                None,   # int  — segment end in source video
        "provisional_emotion":    None,   # str  — from sources YAML, not trusted
        "provisional_style":      None,   # str  — from sources YAML, not trusted
    }


def _empty_pipeline_stages() -> dict:
    """
    One sub-object per pipeline stage.
    Each has: passed (bool|None), reason (str|None), and stage-specific fields.
    None means the stage hasn't run yet.
    """
    return {
        "download": {
            "passed":    None,
            "reason":    None,
            "raw_path":  None,   # str — path to raw WAV
        },
        "preprocess": {
            "passed":       None,
            "reason":       None,
            "cleaned_path": None,   # str — path to normalized WAV
        },
        "segment": {
            "passed":        None,
            "reason":        None,
            "segment_index": None,  # int — which segment within the source clip
            "processed_path": None, # str — path to this segment's WAV
            "duration_sec":  None,  # float
        },
        "quality_check": {
            "passed":         None,
            "reason":         None,
            "snr_db":         None,   # float
            "silence_ratio":  None,   # float
            "clipping_pct":   None,   # float
            "quality_score":  None,   # float  0–1
            "reject_reasons": [],     # list[str]
        },
        "transcription": {
            "passed":       None,
            "reason":       None,
            "transcript":   None,   # str — raw ASR output (draft, not corrected)
            "model":        None,   # str — e.g. "saaras:v3"
            "language_code": None,
        },
        "diarization": {
            "passed":           None,
            "reason":           None,
            "speaker_count":    None,   # int
            "is_single_speaker": None,  # bool
        },
        "emotion_tagging": {
            "passed":             None,
            "reason":             None,
            "emotion":            None,   # str
            "style":              None,   # str
            "confidence":         None,   # float
            "needs_manual_review": None,  # bool
            "model_used":         None,   # str
        },
    }


def _empty_manual_review() -> dict:
    """
    This sub-object is intentionally left empty by the pipeline.
    The human auditor fills it in during the listening pass via run_audit.py.
    Fields:
      listened            — has a human played this clip from start to finish?
      transcript_corrected— hand-corrected transcript; null = ASR output was fine
      emotion_corrected   — hand-corrected emotion tag; null = pipeline tag was fine
      notes               — free text, anything worth recording (e.g. "noisy tail",
                            "speaker switches to English mid-sentence", etc.)
      verdict             — final human decision:
                              "keep"       → include in dataset
                              "reject"     → exclude, never upload
                              "needs_fix"  → fixable but not done yet (revisit)
    """
    return {
        "listened":             False,
        "transcript_corrected": None,
        "emotion_corrected":    None,
        "notes":                "",
        "verdict":              None,   # None = not yet reviewed
    }


def _new_clip_record(clip_id: str) -> dict:
    """Return a fully-structured, all-None manifest record for a new clip."""
    return {
        "id":           clip_id,
        "created_at":   _utcnow(),
        "updated_at":   _utcnow(),
        "source_info":  _empty_source_info(),
        "stages":       _empty_pipeline_stages(),
        "manual_review": _empty_manual_review(),
    }


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# In-memory state + lock
# ---------------------------------------------------------------------------

_manifest: dict[str, dict] = {}   # clip_id → record
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_manifest() -> dict[str, dict]:
    """
    Load manifest.json from disk into memory.
    If the file doesn't exist yet, starts with an empty manifest.
    Always returns the in-memory dict (clip_id → record).

    Call this once at pipeline startup.
    """
    global _manifest

    with _lock:
        if MANIFEST_PATH.exists():
            try:
                raw = MANIFEST_PATH.read_text(encoding="utf-8")
                data = json.loads(raw) if raw.strip() else []

                # Stored as a list (for human readability) — convert to dict
                if isinstance(data, list):
                    _manifest = {rec["id"]: rec for rec in data}
                elif isinstance(data, dict):
                    # Support both {clip_id: record} and {"clips": [...]} shapes
                    if "clips" in data:
                        _manifest = {rec["id"]: rec for rec in data["clips"]}
                    else:
                        _manifest = data
                else:
                    log.warning("[manifest] Unexpected JSON shape — starting fresh.")
                    _manifest = {}

                log.info(f"[manifest] Loaded {len(_manifest)} clips from {MANIFEST_PATH}")

            except (json.JSONDecodeError, KeyError) as e:
                log.error(
                    f"[manifest] Failed to parse {MANIFEST_PATH}: {e}. "
                    "Starting with empty manifest — existing file renamed to .bak"
                )
                _backup_corrupt_manifest()
                _manifest = {}
        else:
            log.info(f"[manifest] {MANIFEST_PATH} not found — starting fresh.")
            MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
            _manifest = {}

    return deepcopy(_manifest)


def update_clip(clip_id: str, **fields: Any) -> dict:
    """
    Merge `fields` into the manifest record for `clip_id`.
    Creates a new record if `clip_id` is not yet in the manifest.

    Supported top-level keyword groups (pass as dicts):
        source_info     — overrides individual source_info fields
        stages          — deep-merged into the stages sub-object
        manual_review   — overrides individual manual_review fields

    Any other keyword argument is set directly on the top-level record.

    Deep merge semantics:
        - Dicts are merged recursively (new keys added, existing keys updated)
        - Lists are replaced, not appended (pass the full new list)
        - None values are written as-is (use None to explicitly clear a field)

    Returns the updated record (a copy).

    Example:
        update_clip(
            "en_001",
            source_info={
                "url": "https://youtu.be/...",
                "language": "en-IN",
                "source_type": "tedx_talk",
            },
            stages={
                "quality_check": {
                    "passed": True,
                    "snr_db": 34.2,
                    "quality_score": 0.91,
                    "reject_reasons": [],
                }
            },
        )
    """
    with _lock:
        if clip_id not in _manifest:
            log.info(f"[manifest] Creating new record for {clip_id}")
            _manifest[clip_id] = _new_clip_record(clip_id)

        record = _manifest[clip_id]
        record["updated_at"] = _utcnow()

        for key, value in fields.items():
            if key in ("source_info", "manual_review"):
                # Shallow merge into the sub-object
                if isinstance(value, dict):
                    record[key].update(value)
                else:
                    log.warning(
                        f"[manifest] Expected dict for '{key}', got {type(value).__name__}. "
                        "Skipping."
                    )

            elif key == "stages":
                # Deep merge: stages → stage_name → fields
                if not isinstance(value, dict):
                    log.warning("[manifest] 'stages' must be a dict. Skipping.")
                    continue
                for stage_name, stage_fields in value.items():
                    if stage_name not in record["stages"]:
                        log.warning(
                            f"[manifest] Unknown stage '{stage_name}' — "
                            "adding anyway (may indicate a pipeline change)."
                        )
                        record["stages"][stage_name] = {}
                    if isinstance(stage_fields, dict):
                        record["stages"][stage_name].update(stage_fields)
                    else:
                        record["stages"][stage_name] = stage_fields

            else:
                # Top-level field (e.g. a custom field added mid-project)
                record[key] = value

        log.debug(f"[manifest] Updated {clip_id}: {list(fields.keys())}")
        return deepcopy(record)


def save_manifest() -> None:
    """
    Write the in-memory manifest to data/manifest.json atomically.

    Uses write-to-temp-file + os.replace() so a crash or KeyboardInterrupt
    mid-write never leaves a half-written or corrupt JSON file on disk.

    The file is stored as a JSON array (list of records) sorted by clip_id
    for human readability and stable diffs in git.
    """
    with _lock:
        records = sorted(_manifest.values(), key=lambda r: r["id"])
        payload = json.dumps(records, indent=2, ensure_ascii=False)

        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Write to a temp file in the same directory, then atomically replace
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=MANIFEST_PATH.parent,
            prefix=".manifest_tmp_",
            suffix=".json",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(payload)
            for attempt in range(5):
                try:
                    os.replace(tmp_path, MANIFEST_PATH)
                    break
                except PermissionError:
                    if attempt == 4:
                        shutil.copyfile(tmp_path, MANIFEST_PATH)
                        os.unlink(tmp_path)
                        break
                    time.sleep(0.2 * (attempt + 1))
            log.debug(f"[manifest] Saved {len(records)} clips to {MANIFEST_PATH}")
        except Exception as e:
            # Clean up the temp file if the replace failed
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise RuntimeError(f"[manifest] Failed to save manifest: {e}") from e


def get_clip(clip_id: str) -> dict | None:
    """Return a copy of a single clip record, or None if not found."""
    with _lock:
        record = _manifest.get(clip_id)
        return deepcopy(record) if record else None


def all_clips() -> list[dict]:
    """Return a snapshot of all clip records as a list, sorted by id."""
    with _lock:
        return sorted(
            (deepcopy(r) for r in _manifest.values()),
            key=lambda r: r["id"],
        )


def clips_by_verdict(verdict: str) -> list[dict]:
    """
    Return all clips where manual_review.verdict == verdict.
    verdict should be one of: "keep", "reject", "needs_fix", None (not yet reviewed).
    """
    with _lock:
        return [
            deepcopy(r)
            for r in _manifest.values()
            if r["manual_review"]["verdict"] == verdict
        ]


def summary() -> dict:
    """
    Return a quick summary dict useful for progress checks and the report.
    """
    with _lock:
        total = len(_manifest)
        verdicts = {"keep": 0, "reject": 0, "needs_fix": 0, "unreviewed": 0}
        languages = {}
        stages_passed = {s: 0 for s in _empty_pipeline_stages()}

        for rec in _manifest.values():
            v = rec["manual_review"]["verdict"]
            if v in verdicts:
                verdicts[v] += 1
            else:
                verdicts["unreviewed"] += 1

            lang = rec["source_info"].get("language") or "unknown"
            languages[lang] = languages.get(lang, 0) + 1

            for stage_name, stage_data in rec["stages"].items():
                if isinstance(stage_data, dict) and stage_data.get("passed") is True:
                    stages_passed[stage_name] += 1

        return {
            "total_clips":    total,
            "verdicts":       verdicts,
            "languages":      languages,
            "stages_passed":  stages_passed,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _backup_corrupt_manifest() -> None:
    """Rename a corrupt manifest to .bak so it's not lost but not loaded."""
    bak_path = MANIFEST_PATH.with_suffix(".json.bak")
    try:
        MANIFEST_PATH.rename(bak_path)
        log.info(f"[manifest] Corrupt manifest backed up to {bak_path}")
    except OSError as e:
        log.warning(f"[manifest] Could not back up corrupt manifest: {e}")


# ---------------------------------------------------------------------------
# Manual test — run as: uv run python -m src.manifest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("\n" + "═" * 60)
    print("  MANIFEST — SMOKE TEST")
    print("═" * 60)

    # Use a temp path so the smoke test doesn't touch real data
    MANIFEST_PATH = Path("data/manifest_test.json")

    load_manifest()

    # Simulate the pipeline writing a clip through all stages
    update_clip(
        "en_001",
        source_info={
            "url":         "https://youtu.be/test",
            "language":    "en-IN",
            "source_type": "tedx_talk",
            "source_title": "Test TED talk",
        },
        stages={
            "download": {"passed": True, "raw_path": "raw/en_001.wav"},
        },
    )

    update_clip(
        "en_001",
        stages={
            "quality_check": {
                "passed":        True,
                "snr_db":        34.2,
                "silence_ratio": 0.08,
                "clipping_pct":  0.001,
                "quality_score": 0.91,
                "reject_reasons": [],
            },
            "transcription": {
                "passed":     True,
                "transcript": "This is a test transcript.",
                "model":      "saaras:v3",
                "language_code": "en-IN",
            },
            "emotion_tagging": {
                "passed":     True,
                "emotion":    "motivational",
                "style":      "motivational",
                "confidence": 0.88,
                "needs_manual_review": False,
            },
        },
    )

    # Simulate a second clip that gets rejected
    update_clip(
        "en_002",
        source_info={"language": "en-IN", "source_type": "news_solo_segment"},
        stages={
            "quality_check": {
                "passed":        False,
                "snr_db":        11.0,
                "reject_reasons": ["snr_below_threshold"],
            },
        },
    )

    save_manifest()
    print(f"\n✅ Manifest saved to {MANIFEST_PATH}")

    # Reload from disk and verify round-trip
    _manifest.clear()
    load_manifest()

    clip = get_clip("en_001")
    assert clip is not None, "en_001 not found after reload"
    assert clip["stages"]["quality_check"]["snr_db"] == 34.2
    assert clip["stages"]["transcription"]["transcript"] == "This is a test transcript."
    assert clip["manual_review"]["verdict"] is None, "manual_review should start empty"

    print("✅ Round-trip reload verified")
    print(f"✅ manual_review is empty (verdict=None) as expected")

    s = summary()
    print(f"\nSummary: {json.dumps(s, indent=2)}")

    assert s["total_clips"] == 2
    assert s["verdicts"]["unreviewed"] == 2

    print("\n✅ All assertions passed")

    # Clean up test file
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()
        print(f"✅ Test file {MANIFEST_PATH} cleaned up")

    print("═" * 60 + "\n")

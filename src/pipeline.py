"""Batch orchestration for the Sarvam TTS dataset pipeline."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable

import yaml
from rich.console import Console

from src.config import PROJECT_ROOT, settings
from src.diarizer import check_single_speaker
from src.downloader import download_segment
from src.emotion_tagger import tag_emotion
from src.manifest import all_clips, load_manifest, save_manifest, update_clip
from src.preprocessor import normalize_audio
from src.quality_checker import check_quality
from src.segmenter import segment_audio
from src.transcriber import transcribe_clip

console = Console()
STAGES = ("download", "preprocess", "segment", "quality", "transcribe", "diarize", "emotion")


def _project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _load_sources(path: Path, language: str, limit: int | None) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sources = data.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError(f"{path} must contain a top-level 'sources' list.")

    normalized: list[dict] = []
    prefix = language.split("-")[0].lower()
    for index, source in enumerate(sources, 1):
        if "REPLACE_ME" in str(source.get("url", "")):
            console.print(f"[yellow]Skipping placeholder source {index} in {path.name}[/yellow]")
            continue
        source = dict(source)
        source["language"] = source.get("language") or language
        source["source_id"] = source.get("id") or f"{prefix}_src{index:03d}"
        normalized.append(source)

    return normalized[:limit] if limit else normalized


def _stage_enabled(stage: str, stage_from: str, stage_to: str = "emotion") -> bool:
    stage_index = STAGES.index(stage)
    return STAGES.index(stage_from) <= stage_index <= STAGES.index(stage_to)


def _extract_segment(input_path: Path, output_path: Path, start_sec: float, end_sec: float) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        str(input_path),
        "-ar",
        str(settings.sample_rate),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg segment extract failed for {output_path}: {result.stderr}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no segment output at {output_path}")
    return output_path


def _records_with_segments() -> list[dict]:
    load_manifest()
    return [
        record for record in all_clips()
        if record["stages"]["segment"].get("processed_path")
    ]


def _audio_path(record: dict) -> Path | None:
    stored = record["stages"]["segment"].get("processed_path")
    if not stored:
        return None
    path = Path(stored)
    candidates = [path] if path.is_absolute() else [PROJECT_ROOT / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _mark_stale_segments(source_id: str) -> None:
    prefix = f"{source_id}_seg"
    for record in all_clips():
        clip_id = record["id"]
        if not clip_id.startswith(prefix):
            continue
        update_clip(
            clip_id,
            stages={
                "segment": {
                    "passed": False,
                    "reason": "stale_after_resegmentation",
                    "processed_path": None,
                    "duration_sec": None,
                }
            },
            manual_review={
                "listened": False,
                "verdict": None,
                "notes": "Stale segment record from an earlier segmentation pass.",
            },
        )


def _iter_stage_records(limit: int | None) -> Iterable[dict]:
    records = _records_with_segments()
    return records[:limit] if limit else records


def run_pipeline(
    sources_file: Path,
    language: str,
    limit: int | None = None,
    stage_from: str = "download",
    stage_to: str = "emotion",
) -> None:
    if STAGES.index(stage_from) > STAGES.index(stage_to):
        raise ValueError(f"stage_from={stage_from!r} must come before stage_to={stage_to!r}.")

    sources_file = _project_path(sources_file)
    sources = _load_sources(sources_file, language, limit)
    raw_dir = _project_path(settings.raw_dir)
    processed_dir = _project_path(settings.processed_dir)

    if not sources and _stage_enabled("download", stage_from, stage_to):
        raise RuntimeError(
            f"No real sources found in {sources_file}. Replace REPLACE_ME placeholders first."
        )

    load_manifest()

    if _stage_enabled("download", stage_from, stage_to):
        for source in sources:
            raw_path = download_segment(
                url=source["url"],
                start_sec=int(source["start_sec"]),
                end_sec=int(source["end_sec"]),
                clip_id=source["source_id"],
                output_dir=raw_dir,
            )
            if raw_path:
                update_clip(
                    source["source_id"],
                    source_info={
                        "url": source["url"],
                        "source_title": source.get("source_title"),
                        "source_type": source.get("source_type"),
                        "language": source["language"],
                        "start_sec": source.get("start_sec"),
                        "end_sec": source.get("end_sec"),
                        "provisional_emotion": source.get("provisional_emotion"),
                        "provisional_style": source.get("provisional_style"),
                    },
                    stages={"download": {"passed": True, "raw_path": _display_path(raw_path)}},
                )
            else:
                update_clip(source["source_id"], stages={"download": {"passed": False, "reason": "download_failed"}})
            save_manifest()

    if _stage_enabled("preprocess", stage_from, stage_to):
        for source in sources:
            source_id = source["source_id"]
            raw_path = raw_dir / f"{source_id}.wav"
            cleaned_path = processed_dir / f"{source_id}_clean.wav"
            try:
                normalize_audio(raw_path, cleaned_path)
                update_clip(source_id, stages={"preprocess": {"passed": True, "cleaned_path": _display_path(cleaned_path)}})
            except Exception as exc:
                console.print(f"[red]{source_id} preprocess failed:[/red] {exc}")
                update_clip(source_id, stages={"preprocess": {"passed": False, "reason": str(exc)}})
            save_manifest()

    if _stage_enabled("segment", stage_from, stage_to):
        for source in sources:
            source_id = source["source_id"]
            cleaned_path = processed_dir / f"{source_id}_clean.wav"
            _mark_stale_segments(source_id)
            ranges = segment_audio(
                cleaned_path,
                min_sec=settings.target_clip_min_sec,
                max_sec=settings.target_clip_max_sec,
            )
            if not ranges:
                update_clip(source_id, stages={"segment": {"passed": False, "reason": "no_segments"}})
                save_manifest()
                continue

            for segment_index, (start, end) in enumerate(ranges):
                clip_id = f"{source_id}_seg{segment_index:02d}"
                out_path = processed_dir / f"{clip_id}.wav"
                try:
                    _extract_segment(cleaned_path, out_path, start, end)
                    update_clip(
                        clip_id,
                        source_info={
                            "url": source["url"],
                            "source_title": source.get("source_title"),
                            "source_type": source.get("source_type"),
                            "language": source["language"],
                            "start_sec": source.get("start_sec"),
                            "end_sec": source.get("end_sec"),
                            "provisional_emotion": source.get("provisional_emotion"),
                            "provisional_style": source.get("provisional_style"),
                        },
                        stages={
                            "download": {"passed": True, "raw_path": _display_path(raw_dir / f"{source_id}.wav")},
                            "preprocess": {"passed": True, "cleaned_path": _display_path(cleaned_path)},
                            "segment": {
                                "passed": True,
                                "segment_index": segment_index,
                                "processed_path": _display_path(out_path),
                                "duration_sec": round(end - start, 3),
                            },
                        },
                    )
                except Exception as exc:
                    update_clip(clip_id, stages={"segment": {"passed": False, "reason": str(exc)}})
                save_manifest()

    if _stage_enabled("quality", stage_from, stage_to):
        for record in _iter_stage_records(limit):
            path = _audio_path(record)
            if not path:
                continue
            result = check_quality(
                path,
                settings.snr_threshold_db,
                settings.silence_ratio_max,
                settings.clipping_max_pct,
            )
            update_clip(record["id"], stages={"quality_check": result})
            save_manifest()

    if _stage_enabled("transcribe", stage_from, stage_to):
        for record in _iter_stage_records(limit):
            path = _audio_path(record)
            if not path or record["stages"]["quality_check"].get("passed") is False:
                continue
            result = transcribe_clip(path, record["source_info"].get("language") or language)
            update_clip(
                record["id"],
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
            save_manifest()

    if _stage_enabled("diarize", stage_from, stage_to):
        for record in _iter_stage_records(limit):
            path = _audio_path(record)
            if not path or record["stages"]["transcription"].get("passed") is False:
                continue
            result = check_single_speaker(path, record["source_info"].get("language") or language)
            update_clip(
                record["id"],
                stages={
                    "diarization": {
                        "passed": result["is_single_speaker"],
                        "reason": result["error"],
                        "speaker_count": result["speaker_count"],
                        "is_single_speaker": result["is_single_speaker"],
                    }
                },
            )
            save_manifest()

    if _stage_enabled("emotion", stage_from, stage_to):
        for record in _iter_stage_records(limit):
            transcript = record["stages"]["transcription"].get("transcript") or ""
            if not transcript:
                continue
            result = tag_emotion(
                transcript=transcript,
                language=record["source_info"].get("language") or language,
                source_type=record["source_info"].get("source_type") or "unknown",
            )
            update_clip(
                record["id"],
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
            save_manifest()

    console.print("[bold green]Pipeline run complete.[/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Sarvam TTS dataset pipeline.")
    parser.add_argument("--sources", type=Path, default=PROJECT_ROOT / "data" / "sources_en.yaml")
    parser.add_argument("--language", default="en-IN")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stage-from", choices=STAGES, default="download")
    parser.add_argument("--stage-to", choices=STAGES, default="emotion")
    args = parser.parse_args()

    run_pipeline(
        sources_file=args.sources,
        language=args.language,
        limit=args.limit,
        stage_from=args.stage_from,
        stage_to=args.stage_to,
    )


if __name__ == "__main__":
    main()

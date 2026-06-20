from __future__ import annotations

from collections import Counter, defaultdict

from rich.console import Console
from rich.table import Table

from src.manifest import all_clips, load_manifest


TARGET_TOTAL_MIN = 60.0
TARGET_PER_LANGUAGE_MIN = 30.0


def _duration(record: dict) -> float:
    return float(record.get("stages", {}).get("segment", {}).get("duration_sec") or 0.0)


def main() -> None:
    console = Console()
    load_manifest()
    records = all_clips()
    segments = [r for r in records if r.get("stages", {}).get("segment", {}).get("processed_path")]
    kept = [r for r in segments if r.get("manual_review", {}).get("verdict") == "keep"]
    reviewed = [r for r in segments if r.get("manual_review", {}).get("verdict")]

    by_language_sec: dict[str, float] = defaultdict(float)
    reviewed_by_language_sec: dict[str, float] = defaultdict(float)
    segment_by_language_sec: dict[str, float] = defaultdict(float)

    for record in segments:
        lang = record.get("source_info", {}).get("language") or "unknown"
        segment_by_language_sec[lang] += _duration(record)
        if record.get("manual_review", {}).get("verdict"):
            reviewed_by_language_sec[lang] += _duration(record)
        if record.get("manual_review", {}).get("verdict") == "keep":
            by_language_sec[lang] += _duration(record)

    verdicts = Counter(r.get("manual_review", {}).get("verdict") or "unreviewed" for r in segments)
    stages = [
        "download",
        "preprocess",
        "segment",
        "quality_check",
        "transcription",
        "diarization",
        "emotion_tagging",
    ]
    stage_counts = {
        stage: sum(1 for r in records if r.get("stages", {}).get(stage, {}).get("passed") is True)
        for stage in stages
    }

    console.rule("[bold cyan]Dataset Status[/bold cyan]")

    overview = Table("Metric", "Value")
    overview.add_row("Manifest records", str(len(records)))
    overview.add_row("Segment records", str(len(segments)))
    overview.add_row("Reviewed segments", f"{len(reviewed)} / {len(segments)}")
    overview.add_row("Kept segments", str(len(kept)))
    overview.add_row("Kept duration", f"{sum(by_language_sec.values()) / 60:.2f} min / {TARGET_TOTAL_MIN:.0f} min")
    console.print(overview)

    lang_table = Table("Language", "Segmented", "Reviewed", "Kept", "Remaining to 30 min")
    languages = sorted(set(segment_by_language_sec) | set(reviewed_by_language_sec) | set(by_language_sec))
    for language in languages:
        kept_min = by_language_sec[language] / 60
        remaining = max(0.0, TARGET_PER_LANGUAGE_MIN - kept_min)
        lang_table.add_row(
            language,
            f"{segment_by_language_sec[language] / 60:.2f} min",
            f"{reviewed_by_language_sec[language] / 60:.2f} min",
            f"{kept_min:.2f} min",
            f"{remaining:.2f} min",
        )
    console.print(lang_table)

    verdict_table = Table("Verdict", "Segments")
    for verdict, count in sorted(verdicts.items()):
        verdict_table.add_row(verdict, str(count))
    console.print(verdict_table)

    stage_table = Table("Stage", "Passed records")
    for stage, count in stage_counts.items():
        stage_table.add_row(stage, str(count))
    console.print(stage_table)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from collections import defaultdict

from rich.console import Console
from rich.table import Table

from src.manifest import all_clips, load_manifest, save_manifest, update_clip


def _duration(record: dict) -> float:
    return float(record.get("stages", {}).get("segment", {}).get("duration_sec") or 0.0)


def _quality_score(record: dict) -> float:
    return float(record.get("stages", {}).get("quality_check", {}).get("quality_score") or 0.0)


def _is_candidate(record: dict) -> bool:
    stages = record.get("stages", {})
    return bool(
        stages.get("segment", {}).get("processed_path")
        and stages.get("quality_check", {}).get("passed") is True
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mark a balanced, high-quality ASR subset in data/manifest.json."
    )
    parser.add_argument("--target-minutes", type=float, default=32.0)
    parser.add_argument("--languages", nargs="+", default=["en-IN", "hi-IN"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    console = Console()
    load_manifest()
    records = all_clips()

    selected_by_language: dict[str, list[dict]] = defaultdict(list)
    for language in args.languages:
        pool = [
            record for record in records
            if record.get("source_info", {}).get("language") == language and _is_candidate(record)
        ]
        pool.sort(key=lambda record: (_quality_score(record), _duration(record)), reverse=True)

        total = 0.0
        for record in pool:
            if total >= args.target_minutes * 60:
                break
            selected_by_language[language].append(record)
            total += _duration(record)

    selected_ids = {
        record["id"]
        for selected in selected_by_language.values()
        for record in selected
    }

    if not args.dry_run:
        for record in records:
            if not _is_candidate(record):
                continue
            is_selected = record["id"] in selected_ids
            note = (
                "Selected for Sarvam ASR/emotion pass."
                if is_selected
                else "Quality-passed reserve clip; not needed for the balanced 60-minute target."
            )
            update_clip(
                record["id"],
                asr_subset=is_selected,
                manual_review={"notes": note if not record.get("manual_review", {}).get("notes") else record["manual_review"]["notes"]},
            )
        save_manifest()

    table = Table("Language", "Selected clips", "Selected duration", "Mean quality")
    for language in args.languages:
        selected = selected_by_language[language]
        total_sec = sum(_duration(record) for record in selected)
        mean_quality = (
            sum(_quality_score(record) for record in selected) / len(selected)
            if selected
            else 0.0
        )
        table.add_row(language, str(len(selected)), f"{total_sec / 60:.2f} min", f"{mean_quality:.3f}")
    console.print(table)

    status = "Dry run only; manifest not changed." if args.dry_run else "Manifest updated with asr_subset flags."
    console.print(f"[green]{status}[/green]")


if __name__ == "__main__":
    main()

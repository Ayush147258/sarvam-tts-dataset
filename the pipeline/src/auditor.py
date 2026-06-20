"""Interactive manual audit pass for manifest clips."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from src.config import PROJECT_ROOT
from src.manifest import all_clips, load_manifest, save_manifest, summary, update_clip

console = Console()
VALID_VERDICTS = {"keep", "reject", "needs_fix", "skip"}


def _resolve_audio_path(record: dict) -> Path | None:
    segment_path = record["stages"]["segment"].get("processed_path")
    candidates: list[Path] = []
    if segment_path:
        path = Path(segment_path)
        candidates.append(path if path.is_absolute() else PROJECT_ROOT / path)
        candidates.append(path)
    candidates.append(PROJECT_ROOT / "processed" / f"{record['id']}.wav")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _play_audio(path: Path) -> None:
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif os.uname().sysname == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        console.print(f"[yellow]Could not auto-open audio:[/yellow] {exc}")
        console.print(f"Open manually: {path}")


def _candidate_records(records: list[dict], include_reviewed: bool, needs_review_only: bool) -> list[dict]:
    candidates = [r for r in records if r["stages"]["segment"].get("processed_path")]
    if not include_reviewed:
        candidates = [r for r in candidates if not r["manual_review"].get("listened")]
    if needs_review_only:
        candidates = [
            r for r in candidates
            if r["stages"]["emotion_tagging"].get("needs_manual_review")
            or r["manual_review"].get("verdict") in (None, "needs_fix")
        ]
    return candidates


def print_summary() -> None:
    load_manifest()
    stats = summary()
    table = Table("Metric", "Value")
    table.add_row("Total clips", str(stats["total_clips"]))
    for verdict, count in stats["verdicts"].items():
        table.add_row(f"Verdict: {verdict}", str(count))
    console.print(table)


def audit_manifest(
    limit: int | None = None,
    include_reviewed: bool = False,
    needs_review_only: bool = False,
    play: bool = True,
) -> None:
    load_manifest()
    records = _candidate_records(all_clips(), include_reviewed, needs_review_only)
    if limit is not None:
        records = records[:limit]

    if not records:
        console.print("[green]No clips need audit with the selected filters.[/green]")
        return

    console.print(f"[bold]Manual audit queue:[/bold] {len(records)} clip(s)")

    for index, record in enumerate(records, 1):
        clip_id = record["id"]
        audio_path = _resolve_audio_path(record)
        transcript = record["stages"]["transcription"].get("transcript") or ""
        emotion = record["stages"]["emotion_tagging"].get("emotion") or "uncertain"
        style = record["stages"]["emotion_tagging"].get("style") or "neutral"
        quality = record["stages"]["quality_check"].get("quality_score")

        console.rule(f"{index}/{len(records)} {clip_id}")
        console.print(f"[bold]Audio:[/bold] {audio_path or 'missing'}")
        console.print(f"[bold]Transcript:[/bold] {transcript or '[missing]'}")
        console.print(f"[bold]Emotion/style:[/bold] {emotion} / {style}")
        console.print(f"[bold]Quality score:[/bold] {quality}")

        if audio_path and play and Confirm.ask("Open audio?", default=True):
            _play_audio(audio_path)

        if not Confirm.ask("Did you listen to the whole clip?", default=True):
            update_clip(clip_id, manual_review={"listened": False, "verdict": None})
            save_manifest()
            continue

        corrected_transcript = Prompt.ask(
            "Corrected transcript (leave blank to keep ASR)",
            default="",
            show_default=False,
        ).strip() or None
        corrected_emotion = Prompt.ask(
            "Corrected emotion (leave blank to keep pipeline tag)",
            default="",
            show_default=False,
        ).strip() or None
        notes = Prompt.ask("Notes", default="", show_default=False).strip()

        verdict = Prompt.ask(
            "Verdict",
            choices=sorted(VALID_VERDICTS),
            default="keep",
        )
        if verdict == "skip":
            console.print("[yellow]Skipped without saving a verdict.[/yellow]")
            continue

        update_clip(
            clip_id,
            manual_review={
                "listened": True,
                "transcript_corrected": corrected_transcript,
                "emotion_corrected": corrected_emotion,
                "notes": notes,
                "verdict": verdict,
            },
        )
        save_manifest()
        console.print(f"[green]Saved audit result for {clip_id}: {verdict}[/green]")

    print_summary()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual listening audit for Sarvam TTS clips.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-reviewed", action="store_true")
    parser.add_argument("--needs-review-only", action="store_true")
    parser.add_argument("--no-play", action="store_true", help="Do not offer to open audio files")
    parser.add_argument("--summary", action="store_true", help="Print audit summary and exit")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    audit_manifest(
        limit=args.limit,
        include_reviewed=args.include_reviewed,
        needs_review_only=args.needs_review_only,
        play=not args.no_play,
    )


if __name__ == "__main__":
    main()

"""
Upload the locally built Sarvam TTS dataset to the HuggingFace Hub.

Run:
    python -m src.upload_hf --repo-id your-name/sarvam-tts-dataset
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import httpx
from datasets import Dataset, DatasetDict, load_from_disk
from huggingface_hub import DatasetCard, HfApi
from rich.console import Console
from rich.logging import RichHandler

from src.config import PROJECT_ROOT, require_hf_token

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger(__name__)
console = Console()

DEFAULT_REPO_ID = "YOUR_HF_USERNAME/sarvam-tts-dataset"
DEFAULT_GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/sarvam-tts-dataset"
DATASET_CACHE_PATH = PROJECT_ROOT / "data" / "dataset_cache"


def _as_dataset(obj: Dataset | DatasetDict) -> Dataset:
    if isinstance(obj, DatasetDict):
        if "train" not in obj:
            raise ValueError("DatasetDict cache has no 'train' split.")
        return obj["train"]
    return obj


def _compute_stats(dataset: Dataset) -> dict:
    total_clips = len(dataset)
    durations = list(dataset["duration_sec"]) if total_clips else []
    total_sec = float(sum(durations))

    lang_counts: dict[str, int] = {}
    for lang in dataset["language"]:
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    emotion_counts: dict[str, int] = {}
    for emotion in dataset["emotion"]:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    n_verified = sum(1 for value in dataset["manually_verified"] if value)
    sorted_durations = sorted(durations)
    n = len(sorted_durations)

    return {
        "total_clips": total_clips,
        "total_sec": total_sec,
        "total_min": total_sec / 60,
        "lang_counts": lang_counts,
        "emotion_counts": emotion_counts,
        "n_verified": n_verified,
        "n_unverified": total_clips - n_verified,
        "pct_verified": (100 * n_verified / total_clips) if total_clips else 0,
        "dur_mean": (total_sec / n) if n else 0,
        "dur_median": sorted_durations[n // 2] if n else 0,
        "dur_min": sorted_durations[0] if n else 0,
        "dur_max": sorted_durations[-1] if n else 0,
    }


def _table(title: str, counts: dict[str, int], total: int) -> str:
    lines = [f"| {title} | Clips | % of total |", "|---|---:|---:|"]
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        pct = 100 * count / total if total else 0
        lines.append(f"| {key} | {count} | {pct:.1f}% |")
    return "\n".join(lines)


def _duration_string(stats: dict) -> str:
    if stats["total_min"] >= 60:
        return f"{stats['total_min'] / 60:.1f} hours ({stats['total_min']:.1f} minutes)"
    return f"{stats['total_min']:.1f} minutes ({stats['total_sec']:.0f} seconds)"


def build_dataset_card(repo_id: str, github_repo_url: str, dataset: Dataset) -> DatasetCard:
    stats = _compute_stats(dataset)
    duration = _duration_string(stats)
    lang_table = _table("Language", stats["lang_counts"], stats["total_clips"])
    emotion_table = _table("Emotion tag", stats["emotion_counts"], stats["total_clips"])

    content = f"""---
language:
  - en
  - hi
license: cc-by-4.0
task_categories:
  - text-to-speech
  - automatic-speech-recognition
pretty_name: Sarvam TTS Indian English and Hindi Dataset
tags:
  - audio
  - speech
  - tts
  - indian-english
  - hindi
  - sarvam
---

# Sarvam TTS Dataset: Indian English and Hindi

A curated speech dataset for TTS model training, containing {stats['total_clips']} clips in Indian English (`en-IN`) and Hindi (`hi-IN`) totalling **{duration}**.

Pipeline source code: [{github_repo_url}]({github_repo_url})

## Dataset Summary

| Stat | Value |
|---|---:|
| Total clips | {stats['total_clips']} |
| Total duration | {duration} |
| Clip duration range | {stats['dur_min']:.1f}s to {stats['dur_max']:.1f}s |
| Mean clip duration | {stats['dur_mean']:.1f}s |
| Median clip duration | {stats['dur_median']:.1f}s |
| Manually verified | {stats['n_verified']} / {stats['total_clips']} ({stats['pct_verified']:.0f}%) |
| Automatically tagged only | {stats['n_unverified']} |

## Language Balance

{lang_table}

## Emotion and Style Tags

{emotion_table}

Valid emotion tags include `neutral`, `happy`, `sad`, `excited`, `angry`, `formal`, `informational`, `storytelling`, `conversational`, `motivational`, `whisper`, `dramatic`, and `uncertain`.

## Manual Verification

Human audit status is stored in the `manually_verified` column. Rows with `manually_verified: false` use automatic ASR and automatic emotion/style tags and should be treated with lower confidence.

## Dataset Schema

| Column | Description |
|---|---|
| `id` | Unique clip identifier |
| `language` | `en-IN` or `hi-IN` |
| `audio` | WAV audio decoded by `datasets.Audio` |
| `transcript` | Manual correction or ASR transcript |
| `duration_sec` | Clip duration in seconds |
| `emotion` | Emotion tag |
| `style` | Delivery style tag |
| `speaker_gender` | `male`, `female`, or `unknown` |
| `source_url` | Original source URL |
| `source_title` | Source title |
| `snr_db` | Estimated SNR |
| `sample_rate` | Expected sample rate |
| `quality_score` | Composite quality score from 0 to 1 |
| `manually_verified` | Whether a human reviewed transcript and tags |
| `created_at` | Processing date |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
row = ds["train"][0]
print(row["transcript"])
print(row["audio"]["sampling_rate"])
```

## Limitations

Emotion and style tags are derived primarily from transcripts, so prosody-only cues such as sarcasm or subdued excitement may need manual correction. Source licensing must be verified before public redistribution.
"""
    return DatasetCard(content)


def upload_dataset(
    repo_id: str,
    dataset_cache_path: Path = DATASET_CACHE_PATH,
    github_repo_url: str = DEFAULT_GITHUB_REPO_URL,
    private: bool = False,
) -> str:
    token = require_hf_token()
    if repo_id == DEFAULT_REPO_ID:
        raise ValueError("Pass --repo-id with your real HuggingFace namespace/dataset name.")
    if not dataset_cache_path.exists():
        raise FileNotFoundError(
            f"Dataset cache not found at {dataset_cache_path}. Run python -m src.dataset_builder first."
        )

    loaded = load_from_disk(str(dataset_cache_path))
    dataset = _as_dataset(loaded)
    if len(dataset) == 0:
        raise ValueError("Dataset cache is empty; refusing to upload an empty dataset.")

    api = HfApi(token=token)
    log.info("[upload_hf] Creating/updating dataset repo %s", repo_id)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    card = build_dataset_card(repo_id, github_repo_url, dataset)
    card.push_to_hub(repo_id, repo_type="dataset", token=token)

    log.info("[upload_hf] Pushing dataset rows and audio files")
    dataset.push_to_hub(repo_id, token=token, private=private)

    url = f"https://huggingface.co/datasets/{repo_id}"
    if not private:
        _verify_public_url(url)
    return url


def _verify_public_url(url: str) -> None:
    for attempt in range(1, 4):
        try:
            response = httpx.get(url, follow_redirects=True, timeout=20)
            if response.status_code == 200:
                log.info("[upload_hf] Public URL reachable: %s", url)
                return
            log.warning("[upload_hf] URL check returned HTTP %s", response.status_code)
        except httpx.HTTPError as exc:
            log.warning("[upload_hf] URL check attempt %s failed: %s", attempt, exc)
        time.sleep(2 * attempt)
    raise RuntimeError(f"Uploaded dataset, but public URL could not be verified: {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the Sarvam TTS dataset to HuggingFace Hub.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HuggingFace dataset repo id, e.g. user/name")
    parser.add_argument("--dataset-cache", type=Path, default=DATASET_CACHE_PATH)
    parser.add_argument("--github-repo-url", default=DEFAULT_GITHUB_REPO_URL)
    parser.add_argument("--private", action="store_true", help="Create/upload as a private dataset")
    args = parser.parse_args()

    url = upload_dataset(
        repo_id=args.repo_id,
        dataset_cache_path=args.dataset_cache,
        github_repo_url=args.github_repo_url,
        private=args.private,
    )
    console.print(f"[bold green]Uploaded dataset:[/bold green] {url}")


if __name__ == "__main__":
    main()

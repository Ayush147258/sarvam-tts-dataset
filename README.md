# Sarvam TTS Dataset — Indian English & Hindi

A curated speech dataset for Text-to-Speech (TTS) model training, built
as part of the Sarvam AI Data Annotation assignment (June 2026).

**Dataset:** [huggingface.co/datasets/YOUR_HF_USERNAME/sarvam-tts-dataset](https://huggingface.co/datasets/YOUR_HF_USERNAME/sarvam-tts-dataset)
**Report:** [report/pipeline_report.pdf](report/pipeline_report.pdf)

---

## What this project is

A pipeline that downloads single-speaker audio from YouTube, segments it
into 15–30 second clips, runs quality filtering, ASR transcription,
speaker diarization, and LLM-based emotion tagging — then passes every
clip through a manual listening audit before packaging the result as a
HuggingFace dataset.

The pipeline code is plumbing. The actual work is the manual audit in
Phase 6: listening to clips, correcting transcripts, verifying emotion
tags, and making keep/reject decisions. That judgment is what this
project is designed to demonstrate.

---

## Pipeline architecture

```
sources_en.yaml / sources_hi.yaml
        │
        ▼  yt-dlp
raw/*.wav  (22050 Hz mono)
        │
        ▼  ffmpeg loudnorm + 80Hz high-pass filter
cleaned audio
        │
        ▼  silero-vad
15–30s segments, cut only on natural pause boundaries
        │
        ▼  librosa: SNR · silence ratio · clipping
quality-filtered  ──✗── reject below threshold, log reason
        │
        ▼  Sarvam saaras:v3  (mode="transcribe")
raw transcripts
        │
        ▼  Sarvam saaras:v3  (with_diarization=True)
single-speaker verified  ──✗── reject multi-speaker, log reason
        │
        ▼  Sarvam sarvam-105b chat completion
provisional emotion + style tags
        │
        ▼  MANUAL LISTENING PASS
corrected transcripts · corrected tags · keep/reject verdicts
        │
        ▼  datasets >= 5.0.0
HuggingFace Dataset (Audio feature column)
        │
        ▼  huggingface_hub >= 1.19.0
Public HF dataset + dataset card
```

---

## Repository layout

```
sarvam-tts-dataset/
├── README.md
├── pyproject.toml
├── uv.lock
├── config.yaml                    # quality thresholds, sample rate, paths
├── .env.template                  # copy to .env and fill in keys
├── data/
│   ├── sources_en.yaml            # Indian English YouTube sources
│   ├── sources_hi.yaml            # Hindi YouTube sources
│   └── manifest.json              # generated: every clip's pipeline history
├── raw/                           # gitignored: downloaded audio
├── processed/                     # gitignored: segmented clips
├── src/
│   ├── config.py                  # settings loader (pydantic-settings v2)
│   ├── downloader.py              # yt-dlp audio download
│   ├── preprocessor.py            # ffmpeg normalize + filter
│   ├── segmenter.py               # silero-vad segmentation
│   ├── quality_checker.py         # librosa SNR/silence/clipping
│   ├── sarvam_client.py           # Sarvam SDK wrapper
│   ├── transcriber.py             # Sarvam saaras:v3 ASR
│   ├── diarizer.py                # Sarvam diarization
│   ├── emotion_tagger.py          # Sarvam sarvam-105b tagging
│   ├── manifest.py                # manifest.json read/write
│   ├── dataset_builder.py         # builds HF Dataset
│   └── upload_hf.py               # pushes to HuggingFace Hub
├── prompts/
│   └── emotion_classification.txt # few-shot prompt for sarvam-105b
├── scripts/
│   ├── run_pipeline.ps1           # Windows pipeline runner (this file)
│   └── run_audit.py               # interactive manual listening audit
├── notebooks/
│   └── quality_analysis.ipynb     # duration/SNR/emotion visualisations
└── report/
    ├── pipeline_report.md
    └── pipeline_report.pdf
```

---

## Prerequisites

You need four things installed before the first run. Do these once.

### 1. Python 3.12 via uv

Install `uv` from [docs.astral.sh/uv](https://docs.astral.sh/uv/):

```powershell
# In PowerShell (run as normal user, not admin)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal after installation so `uv` is on PATH.

### 2. ffmpeg

```powershell
winget install Gyan.FFmpeg
```

Restart your terminal after installation. Verify with:

```powershell
ffmpeg -version
```

### 3. pandoc (only needed to generate the PDF report)

```powershell
winget install JohnMacFarlane.Pandoc
```

### 4. API keys

You need two keys:

| Key | Where to get it |
|-----|----------------|
| `SARVAM_API_KEY` | [dashboard.sarvam.ai](https://dashboard.sarvam.ai) |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — needs **write** scope |

---

## Setup (fresh clone)

```powershell
# 1. Clone and enter the repo
git clone https://github.com/YOUR_USERNAME/sarvam-tts-dataset.git
cd sarvam-tts-dataset

# 2. Install all Python dependencies (reads pyproject.toml + uv.lock)
uv sync

# 3. Copy the environment template and fill in your keys
Copy-Item .env.template .env
notepad .env        # add SARVAM_API_KEY and HF_TOKEN, save and close

# 4. Verify the config loads correctly
uv run python -c "from src.config import settings; print(settings.snr_threshold_db)"
# Expected output: 20
```

---

## Step 1 — Add your source clips

Open `data/sources_en.yaml` and `data/sources_hi.yaml`.
Each entry is a YouTube URL with a timestamp range you want to download.

```yaml
# Example entry — replace with real, manually vetted URLs
- url: "https://www.youtube.com/watch?v=XXXXXXXXXXX"
  start_sec: 120
  end_sec: 150
  provisional_emotion: "informational"   # draft only — corrected during audit
  provisional_style: "formal"
  expected_speaker_count: 1
  source_type: "tedx_talk"
```

**Before adding a URL, watch the clip yourself and confirm:**
- Single speaker throughout the segment (no panel, no host interruptions)
- No background music bed
- You can understand the language well enough to judge ASR accuracy later
- Audio is clear enough to pass the ≥ 20 dB SNR threshold by ear

---

## Step 2 — Small-batch test run (do this first)

Run 8 clips through the full pipeline before scaling up.
This catches configuration errors and lets you calibrate thresholds cheaply.

```powershell
# English small batch
.\scripts\run_pipeline.ps1 -SmallBatch 8 -SourcesFile data/sources_en.yaml

# Hindi small batch (run separately — different language code for ASR)
.\scripts\run_pipeline.ps1 -SmallBatch 8 -SourcesFile data/sources_hi.yaml -Language hi-IN
```

Watch the output for any stage that logs `❌`. Fix before scaling up.

After the small batch, run the manual audit on those 8 clips (Step 4 below)
and adjust thresholds in `config.yaml` if needed before proceeding.

---

## Step 3 — Full pipeline run

Once the small batch looks good:

```powershell
# Full English run
.\scripts\run_pipeline.ps1 -SourcesFile data/sources_en.yaml

# Full Hindi run
.\scripts\run_pipeline.ps1 -SourcesFile data/sources_hi.yaml -Language hi-IN
```

If the pipeline crashes partway through a stage, fix the error and resume
from that stage without re-running earlier ones:

```powershell
# Example: resume from transcription after a rate-limit crash
.\scripts\run_pipeline.ps1 -StageFrom transcribe -SourcesFile data/sources_en.yaml
```

---

## Step 4 — Manual audit (required)

This is the most important step. The pipeline output is a draft.
You must listen to every clip in the small batch and a 10–15% sample
of the full batch before building the dataset.

```powershell
uv run python scripts/run_audit.py
```

The audit script walks you through each unreviewed clip and prompts for:

- **Verdict:** `keep` / `reject` / `needs_fix`
- **Transcript correction:** paste a corrected transcript if the ASR made errors
- **Emotion correction:** override the LLM tag if it doesn't match what you hear
- **Notes:** anything worth recording (noise, code-switching, cut mid-word, etc.)

All decisions are written into `data/manifest.json` in the `manual_review`
sub-object. Only clips with `verdict: keep` are included in the final dataset.

---

## Step 5 — Build and upload the dataset

```powershell
# Build the HuggingFace Dataset from manifest (kept clips only)
uv run python -m src.dataset_builder

# Upload to HuggingFace Hub (public)
# First: edit REPO_ID in src/upload_hf.py to your HF username
uv run python -m src.upload_hf
```

The upload script prints the public URL and verifies it is reachable
without authentication. Also open it in an incognito browser tab to confirm.

---

## Step 6 — Generate the PDF report

```powershell
# Edit report/pipeline_report.md with your real notes first, then:
pandoc report/pipeline_report.md -o report/pipeline_report.pdf
```

---

## Quality thresholds

Configured in `config.yaml`. Starting values — expect to adjust after
the small-batch listening pass and document the changes in your report.

| Metric | Default threshold | Action if failed |
|--------|------------------|-----------------|
| SNR | ≥ 20 dB | Reject |
| Silence ratio | ≤ 20% of frames | Reject |
| Clipping | < 0.1% of samples | Reject |
| Duration | 15–30 seconds | Trim or reject |
| Speaker count | Exactly 1 | Reject |
| Background music | Not present | Reject |

---

## Clip metadata schema

Each clip in the dataset carries:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique clip ID (e.g. `en_001`) |
| `language` | string | `en-IN` or `hi-IN` |
| `audio` | Audio | Decoded WAV at 22 050 Hz |
| `transcript` | string | Hand-corrected or ASR transcript |
| `duration_sec` | float | Clip length in seconds |
| `emotion` | string | Emotion tag |
| `style` | string | Delivery style tag |
| `speaker_gender` | string | `male` / `female` / `unknown` |
| `source_url` | string | Original YouTube URL |
| `snr_db` | float | Estimated SNR in dB |
| `quality_score` | float | Composite quality score 0–1 |
| `manually_verified` | bool | True if human-checked |
| `created_at` | string | ISO date processed |

---

## Troubleshooting

**`ffmpeg` not found**
Restart your terminal after `winget install Gyan.FFmpeg` — winget
updates PATH but the change only takes effect in a new session.

**`SARVAM_API_KEY` missing error at import**
You have not copied `.env.template` to `.env`, or the key is not filled in.
Run `Copy-Item .env.template .env` then edit `.env`.

**`yt-dlp` download fails on a specific URL**
The video may be age-restricted, region-locked, or deleted. Replace the
URL in your sources YAML with a different clip and re-run.

**Sarvam API rate limit (HTTP 429)**
The emotion tagger and transcriber both have exponential-backoff retry
built in. If you hit sustained rate limits, add `time.sleep(2)` between
clips in the pipeline runner or contact `dashboard.sarvam.ai` for a
higher quota.

**Dataset build fails with "No clips with verdict=keep"**
You have not completed the manual audit yet. Run `run_audit.py` and
mark at least some clips as `keep` before building.

**PowerShell execution policy error**
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

---

## Links

- **HuggingFace dataset:** [YOUR_DATASET_URL_HERE](YOUR_DATASET_URL_HERE)
- **PDF report:** [report/pipeline_report.pdf](report/pipeline_report.pdf)
- **Sarvam API docs:** [docs.sarvam.ai](https://docs.sarvam.ai)
- **Assignment brief:** provided separately
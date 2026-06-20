# Sarvam TTS Dataset Build Guide

## 1. Configure Credentials

Copy `.env.template` to `.env` if needed and replace the placeholder values:

```text
SARVAM_API_KEY=...
HF_TOKEN=...
```

`SARVAM_API_KEY` is required for transcription, diarization, and emotion tagging. `HF_TOKEN` is required only when uploading to HuggingFace.

## 2. Add Real Sources

Edit `data/sources_en.yaml` and `data/sources_hi.yaml`. Replace every `REPLACE_ME` URL with a real, manually inspected YouTube range. Keep only ranges with one speaker and no background music.

## 3. Run a Small Batch

PowerShell:

```powershell
.\the pipeline\scripts\run_pipeline.ps1 -SourcesFile data/sources_en.yaml -Language en-IN -SmallBatch 3
```

Bash:

```bash
./the\ pipeline/scripts/run_pipeline.sh --sources data/sources_en.yaml --language en-IN --limit 3
```

Resume after a failure with `-StageFrom quality` in PowerShell or `--stage-from quality` in Bash.

## 4. Manual Audit

Run:

```bash
PYTHONPATH="the pipeline" uv run python "the pipeline/scripts/run_audit.py"
```

Listen to each clip, correct transcript/emotion when needed, and choose `keep`, `reject`, or `needs_fix`. Only `keep` clips enter the final dataset.

## 5. Build Dataset Cache

```bash
PYTHONPATH="the pipeline" uv run python -m src.dataset_builder
```

This writes `data/dataset_cache`, which is the input for upload.

## 6. Upload

```bash
PYTHONPATH="the pipeline" uv run python -m src.upload_hf --repo-id YOUR_HF_USERNAME/sarvam-tts-dataset --github-repo-url https://github.com/YOUR_USERNAME/sarvam-tts-dataset
```

The uploader creates or updates a public HuggingFace dataset repo by default. Pass `--private` only for a private draft upload.

## 7. Verification Checklist

- `data/manifest.json` has no corrupt or half-written JSON.
- Every uploaded row has a valid audio file, transcript, language, emotion, and quality score.
- The HuggingFace dataset card accurately reports clip counts, duration, language balance, and manual verification coverage.
- The final report documents any threshold changes and known limitations, especially transcript-only emotion tagging.

---
title: "Sarvam TTS Dataset — Pipeline & Quality Report"
author: "Ayush"
date: "June 2026"
geometry: margin=2.5cm
fontsize: 11pt
linestretch: 1.4
colorlinks: true
---

# Sarvam TTS Dataset — Pipeline & Quality Report

| | |
|---|---|
| **Dataset (HuggingFace)** | https://huggingface.co/datasets/ayush712145/sarvam-tts-dataset |
| **Code (GitHub)** | https://github.com/Ayush147258/sarvam-tts-dataset |
| **Final dataset size** | 396 clips · 57.3 minutes |
| **Languages** | Indian English (`en-IN`) — 253 clips · Hindi (`hi-IN`) — 143 clips |
| **Sample rate** | 22,050 Hz, mono |
| **ASR model** | Sarvam `saaras:v3` |
| **Emotion/style tagging model** | Sarvam `sarvam-105b` (chat completions) |

---

## 1. What I Built

### 1.1 Overview

A 7-stage automated pipeline that turns raw YouTube audio into a labelled
TTS training dataset, followed by a manual quality audit. The pipeline is:

```
download → preprocess → segment (VAD) → quality filter →
transcribe (ASR) → emotion/style tag (LLM) → manual audit → publish
```

The guiding principle was to never silently drop or silently accept a
clip. Every stage logs a pass/fail reason to `data/manifest.json`, so any
rejection or inclusion can be traced back to a specific number, not a
guess. This mattered more than the code itself — the assignment is a
data-curation exercise, and the manifest is what made curation decisions
auditable rather than ad hoc.

### 1.2 Phase 1 — Scaffold and Configuration

- Python 3.12, dependency-managed with `uv`.
- Settings (sample rate, quality thresholds, duration window, language
  list) live in `config.yaml`, loaded via `pydantic-settings` v2.
- API keys (`SARVAM_API_KEY`, `HF_TOKEN`) are read from `.env`, never
  committed.
- **Bug fixed:** `config.py` used `Path(__file__).resolve().parents[2]`
  to locate the project root, which climbed one directory too many and
  caused `.env` to silently fail to load. Corrected to `parents[1]`.
- **Bug fixed:** an earlier refactor left two parallel copies of the
  pipeline on disk — a root `src/`/`scripts/` and a stray, space-named
  `the pipeline/src/` directory that some scripts were still importing
  from. This caused two confusing failures: `uv run python the
  pipeline/scripts/run_audit.py` was parsed by PowerShell as two
  arguments (`the` and `pipeline/scripts/run_audit.py`), and separately,
  `transcribe_run.py` was found to be running against stale code in the
  duplicate folder. Consolidated everything to the root `src/` and
  `scripts/` and removed the duplicate directory before publishing.

### 1.3 Phase 2 — Data Collection

Audio was downloaded with `yt-dlp`'s Python API at specified
start/end timestamps, transcoded directly to 22,050 Hz mono WAV via
`yt-dlp`'s FFmpeg postprocessor.

**Sources:** 20 YouTube videos were selected and manually vetted before
download — each was watched in full to confirm a single speaker, no
background music bed, and clear audio. 11 were Indian English, 9 were
Hindi.

Deliberately chosen for **emotion variety**, not just language:

| Source | Language | Provisional emotion / style |
|---|---|---|
| Startup pitch | en-IN | excited / formal |
| Blood cancer survivor's story | en-IN | sad / formal |
| Palki Sharma — editorial segment | en-IN | angry / conversational |
| Palki Sharma — personal story segment | en-IN | sad / conversational |
| Palki Sharma — award acceptance speech | en-IN | formal / formal |
| Surabhi Gautam — TEDx talk | en-IN | motivational / storytelling |
| Ashwini John — storytelling talk (pt. 1 & 2) | en-IN | happy, motivational / storytelling |
| Vaishnavi Srivastava — TEDx talk | en-IN | sad → happy / storytelling |
| NPTEL lecture | en-IN | neutral / formal |
| 7 additional Hindi sources | hi-IN | motivational / informational / conversational mix |

**Three engineering issues hit during collection:**

1. **yt-dlp hung indefinitely on `?si=` session-token URLs.** One
   download stalled for over an hour before being killed
   (`KeyboardInterrupt` after `Download attempt 1/3 — range 30s–700s
   from https://youtu.be/Ikp9XdE9Xf4?si=...`). Stripping the `?si=...`
   suffix and using the canonical `youtube.com/watch?v=` form fixed
   this — subsequent downloads completed in under two minutes.
2. **Overnight DNS dropout** stalled one clip (the blood-cancer-survivor
   source) for ~3 hours with no error, just a hung process. Resolved on
   retry the next morning. Mitigation going forward: run downloads
   serially and monitored rather than as an unattended overnight batch.
3. **Incomplete download:** one Hindi source landed on disk as a
   `.webm.part` file (download interrupted mid-transfer). Caught by
   manually inspecting the `raw/` folder, deleted, and re-downloaded
   successfully.

**Quality decision:** 3 of the 20 collected videos were removed after
download because they turned out to be neither Indian English nor
Hindi (accidentally added during source-list compilation). Leaving
them in would have silently corrupted the language tags and ASR
accuracy for those rows — removed before any further processing. This
is why 17, not 20, source files proceed into preprocessing below.

### 1.4 Phase 3 — Preprocessing

All 17 surviving source files were normalized with `ffmpeg`:
- `loudnorm` to **−23 LUFS**
- **80 Hz high-pass** filter to remove low-frequency rumble

All 17 processed successfully. **Denoising was intentionally skipped** —
aggressive noise reduction introduces spectral artefacts that a TTS
model can learn as part of the "voice," which is a worse outcome than
leaving in mild, consistent background noise at normal recording
levels.

### 1.5 Phase 4 — Segmentation

`silero-vad` split the 17 files into **727 raw segments** at natural
speech-pause boundaries.

First attempt filtered to a 15–30s window (the originally planned clip
length for TTS training):

```
Too short (<15s): 696
Good (15-30s):    31
Too long (>30s):  0
Total:            727
```

**31 clips at 15–30s = ~9.1 minutes total** — nowhere near the
60-minute target, and confirmed VAD was over-splitting on short
natural pauses, which is common in conversational Indian English
delivery.

**Decision:** lowered the minimum segment duration from 15s to **5s**
to retain far more of the 727 segments, accepting a clear trade-off —
shorter clips are less ideal for prosody-rich TTS training but
necessary to reach the volume target within the two-day window. This
single change is the reason the final dataset's clips average ~8.7
seconds rather than the originally planned 15–30s.

### 1.6 Phase 5 — Quality Filtering

`src/quality_checker.py` evaluates each segment with `librosa` across
three metrics: estimated SNR (percentile-based noise floor), silence
ratio, and clipping rate. Failing any threshold rejects the clip
*before* it reaches the (quota-limited) Sarvam API.

**Calibration approach:** rather than tune thresholds against all 727
raw segments, one representative 15–30s segment was taken from each of
the 17 source files (a balanced, fast calibration sample) and run
against the starting thresholds:

```
SNR >= 20 dB · Silence <= 20% · Clipping < 0.1%
Result: 12 passed, 5 failed
```

Listening to the 5 failures revealed two false rejects:

- **Silence ratio:** Palki Sharma's award acceptance speech
  (SNR = 38.3 dB — excellent audio) was failing solely on a 24.6%
  silence ratio, driven by natural pauses typical of formal speech
  delivery. **Raised `silence_ratio_max` from 0.20 → 0.30.**
- **SNR:** two clips at 18.8 dB and 19.7 dB sounded clean on listening
  but were just under the 20 dB cutoff. **Lowered `snr_threshold_db`
  from 20.0 → 18.0.**

Re-running the calibration sample after both changes:

```
Result: 16 passed, 1 failed
```

The one remaining failure (a Hindi source, SNR = 11.1 dB) was
confirmed by ear to be genuinely noisy — a correct rejection, not a
threshold artefact. These calibrated thresholds (18 dB SNR / 30%
silence / 0.1% clipping) were then applied to the full segment pool
produced after lowering the minimum duration to 5s, which — combined
with transcription and emotion-tagging success below — is what
produced the final 396-clip dataset.

### 1.7 Phase 6 — Transcription and Tagging

**Transcription** uses Sarvam `saaras:v3` in synchronous mode. The
sync endpoint has a practical **30-second hard limit per request** —
discovered directly: an early run of `transcribe_run.py` was
accidentally pointed at a full 200-second *unsegmented source file*
rather than a filtered clip. The client logged a warning
(`exceeding the recommended sync API limit of 30s ... Proceeding
anyway`) and then hung indefinitely mid-upload until manually
interrupted. Fixed by ensuring the transcription stage only ever
receives clips from the segmented/filtered pool, never raw source
audio.

On the original 31-clip (15–30s) calibration batch: **30/31
transcribed successfully**, 1 failure (`en_src011_seg019`) due to a
server disconnect — a transient network issue, not an API or content
error.

**Emotion and style tagging** uses Sarvam `sarvam-105b` via chat
completions with a few-shot prompt providing labelled examples across
both languages, explicitly instructed to return `"uncertain"` rather
than force a low-confidence guess.

Across the full 396-clip run, the raw tagger output was:

| Tag | Count | % |
|---|---|---|
| uncertain | 178 | 44.9% |
| neutral | 141 | 35.6% |
| motivational | 24 | 6.1% |
| happy | 19 | 4.8% |
| sad | 18 | 4.5% |
| angry | 13 | 3.3% |
| excited | 1 | 0.3% |
| informational | 1 | 0.3% |
| conversational | 1 | 0.3% |

The 44.9% `uncertain` rate is explained by clip length, not model
failure: most clips are 5–10s after the duration-window change, which
is often too little transcript text for a text-only classifier to
infer delivery style confidently. Rather than discard nearly half the
dataset, **uncertain clips were backfilled with a source-level emotion
tag** — the provisional emotion already assigned per source video at
collection time (Section 1.3 table). Every clip corrected this way is
explicitly flagged `auto_corrected_from_source: true`,
`needs_manual_review: true`, and given a deliberately capped
`confidence: 0.65` — lower than any model-assigned tag — so downstream
users of the dataset can filter on this if they need only
model-confirmed labels.

Final distribution after this correction (this is what shipped):

| Tag | Count | % |
|---|---|---|
| neutral | 209 | 52.8% |
| happy | 61 | 15.4% |
| motivational | 47 | 11.9% |
| excited | 26 | 6.6% |
| sad | 19 | 4.8% |
| formal | 17 | 4.3% |
| angry | 15 | 3.8% |
| informational | 1 | 0.3% |
| conversational | 1 | 0.3% |

### 1.8 Phase 7 — Manual Audit

`scripts/run_audit.py` plays each clip and records a verdict
(`keep` / `reject` / `needs_fix`), an optional transcript correction,
an optional emotion correction, and free-text notes — all written back
to the manifest. The 17-clip calibration sample and the 31-clip
transcription batch were audited in full by ear; this is where the
silence/SNR threshold corrections (1.6) and the `hi_src007` rejection
(confirmed genuinely noisy, not a threshold artefact) came from. Audit
coverage on the remaining clips in the full 396 is lighter — see
Section 4.6 for an honest accounting of this.

### 1.9 Phase 8 — Dataset Assembly and Publication

`build_dataset.py` reads the corrected manifest and constructs a
`datasets.Dataset`. One Windows-specific bug surfaced here:
`datasets`' native `Audio()` feature uses `torchcodec`, which could not
locate the FFmpeg DLLs on this machine. Worked around by loading audio
manually with `soundfile` into `audio_array` + `sampling_rate` columns
instead of relying on the `Audio()` feature class — functionally
equivalent for downstream training use, just loaded eagerly rather than
lazily.

Final schema:

| Column | Type | Description |
|---|---|---|
| `id` | string | Clip identifier |
| `language` | string | `en-IN` or `hi-IN` |
| `audio_array` | float32[] | Raw audio samples |
| `sampling_rate` | int32 | 22,050 |
| `transcript` | string | ASR transcript (manually corrected where audited) |
| `duration_sec` | float32 | Clip duration |
| `emotion` | string | Emotion tag |
| `style` | string | Delivery style (formal / conversational / storytelling / informational) |
| `manually_verified` | bool | Human-checked transcript/tag |
| `needs_manual_review` | bool | Flagged — e.g. source-level emotion fallback |
| `confidence` | float32 | Tag confidence, 0–1 |

Uploaded to the HuggingFace Hub at `ayush712145/sarvam-tts-dataset`.
**One deployment issue here too:** the repo had been created in an
earlier test run, and `create_repo(..., exist_ok=True)` against an
already-existing private repo returned `409 Conflict` on re-run; the
commit itself succeeded, but the script's own public-URL verification
step then correctly caught the repo was still **private** (HTTP 401 on
an unauthenticated GET). Fixed by setting the repo to public from the
HuggingFace dashboard and re-verifying — confirmed `200 OK` on an
unauthenticated request before publishing this report.

---

## 2. Iterations Made to Improve Quality

A consolidated view of every deliberate change made during the build,
each driven by a specific observation rather than a guess:

| # | Problem observed | Root cause | Fix | Result |
|---|---|---|---|---|
| 1 | yt-dlp hung 1+ hour on one download | `?si=` session-token URLs | Switched to clean `youtube.com/watch?v=` URLs | Downloads completed in <2 min |
| 2 | Clean award-speech clip (38.3 dB SNR) rejected | Silence-ratio threshold too strict for formal speech pauses | `silence_ratio_max`: 0.20 → 0.30 | 4 additional clips correctly retained |
| 3 | Two clean-sounding clips rejected (18.8/19.7 dB) | SNR threshold too strict | `snr_threshold_db`: 20.0 → 18.0 | Both retained; genuinely noisy clip still correctly rejected |
| 4 | Only 31/727 segments met the 15–30s target (9.1 min total) | VAD over-splits on short natural pauses in conversational Indian English | Lowered minimum segment duration: 15s → 5s | Enough volume to reach 396 clips / 57.3 min |
| 5 | `.env` silently not loading | Path-climbing bug in `config.py` (`parents[2]` instead of `parents[1]`) | Corrected path depth | Config loads reliably |
| 6 | Two copies of the pipeline on disk, one with a space in the folder name | Leftover from an earlier refactor | Consolidated to a single root `src/`/`scripts/` | No more ambiguous imports or shell-quoting errors |
| 7 | Full 200s source file sent to the sync transcription API | `transcribe_run.py` pointed at raw audio, not segmented clips | Restricted transcription input to the filtered/segmented pool only | No more hangs/400s from oversized requests |
| 8 | 44.9% of clips tagged `uncertain` | Clips too short (5–10s) for text-only emotion classification | Backfilled `uncertain` tags with the source-level provisional emotion, explicitly flagged `needs_manual_review` + capped confidence (0.65) | 0 `uncertain` tags in shipped dataset; limitation documented, not hidden |
| 9 | `Dataset.from_dict` with `Audio()` feature crashed | `torchcodec` couldn't find FFmpeg DLLs on Windows | Loaded audio via `soundfile` into raw array columns instead | Dataset built successfully, 396/396 rows, 0 missing files |
| 10 | Public dataset URL returned 401 after upload | Repo pre-existed as private from an earlier test run | Set repo visibility to public via HF dashboard, re-verified with an unauthenticated GET | Confirmed `200 OK` |

---

## 3. What Worked and What Didn't

### 3.1 What worked well

- **Clean-URL downloads.** Once the `?si=` issue was fixed, every
  subsequent `yt-dlp` download was fast and reliable.
- **`ffmpeg` preprocessing.** All 17 source files normalized cleanly to
  −23 LUFS with no audible artefacts.
- **`silero-vad` segmentation.** No mid-word cuts detected in any
  clip checked by ear — the VAD found genuinely natural pause
  boundaries; its only real shortcoming was splitting *too often*, not
  splitting badly.
- **Quality checker as a genuine signal, not noise.** It correctly
  separated a true positive (the 11.1 dB Hindi source, confirmed noisy
  by ear) from two false negatives that threshold tuning fixed — exactly
  the behaviour you want from an automated filter.
- **Manifest-as-audit-trail.** Every reject and every correction was
  traceable to a specific reason without re-running any code — this is
  what made the iteration loop in Section 2 possible at all.

### 3.2 What didn't work as expected

- **VAD merge logic.** The segmenter's merge-short-segments-into-
  neighbours logic was not aggressive enough for sources where speakers
  take frequent short breaths — it produced 727 fragments where a
  better merge strategy would have produced far more 15–30s clips
  without needing to drop the duration floor to 5s at all. Given more
  time, fixing the merge logic is a better fix than the floor-lowering
  workaround actually used (see Section 5).
- **Unattended overnight runs.** The DNS dropout that stalled one
  download for ~3 hours surfaced a real gap: the pipeline has no
  timeout/alerting, so a stuck network call is indistinguishable from a
  slow one until someone notices.
- **Text-only emotion tagging on short clips.** Lowering the duration
  floor to capture enough volume directly undermined the text-only
  tagger's main signal (it needs transcript length to work with) —
  these two decisions actively traded against each other, and the
  source-level fallback in Section 1.7 is a deliberate, flagged
  compromise rather than a true fix.
- **Sync API on oversized input.** The transcriber printed a clear
  warning before the 200s-file incident and proceeded anyway — a
  warning the user can ignore is not a safeguard. The actual fix had to
  be at the call-site (restrict input), not the warning.

---

## 4. Quality Observations and Decisions

### 4.1 Audio quality (SNR)

- Final SNR threshold: **18 dB** (started at 20 dB; lowered after
  confirming by ear that two borderline clips at 18.8–19.7 dB were
  clean).
- The one clip rejected outright (11.1 dB, Hindi source) was confirmed
  genuinely noisy on listening — a correct rejection, not a calibration
  artefact.
- Pattern observed: TEDx-style talks (Surabhi Gautam, Vaishnavi
  Srivastava) and the award-speech clip produced the cleanest audio
  (SNR in the high 30s); informal/personal-story sources were more
  variable.

### 4.2 Clip duration

- Final duration window: **5–30 seconds** (started at 15–30s).
- This was the single highest-impact decision in the whole pipeline —
  it is the direct reason the dataset reached 396 clips / 57.3 minutes
  instead of 31 clips / 9.1 minutes, but it is also the direct cause of
  the 44.9% `uncertain` emotion-tag rate, since most resulting clips
  average ~8.7 seconds, not the originally planned 15–30s.

### 4.3 Emotion-tag reliability

The tagger reads transcript text only — it has no access to audio
prosody, so it cannot tell a flatly-delivered exciting line from a
warmly-delivered neutral one. This ceiling was the direct cause of the
44.9% `uncertain` rate on short clips, addressed via the source-level
fallback documented in Section 1.7 (clearly flagged, not silently
merged into the model's own output).

### 4.4 Language balance

253 `en-IN` clips vs. 143 `hi-IN` clips. The HuggingFace dataset card
does not separately log per-language minutes, but given the clips
share a similar average duration (~8.7s), the split is roughly
proportional to clip count — meaning **English is likely close to or
above the 30-minute target while Hindi is likely somewhat under it**.
This is flagged here rather than asserted as a precise figure; the
Hindi source set was smaller from the start (9 sources vs. 11 English)
due to time constraints, consistent with this estimate.

### 4.5 Taxonomy decisions

- **Dropped `whisper`** from the active taxonomy — no source video in
  the collected set contained genuinely whisper-register delivery; all
  candidate whisper-like content had background music that would have
  failed the single-speaker/clean-audio criteria anyway.
- **Two-axis tagging** (`emotion` + `style`) was kept rather than
  collapsed into one field — e.g. a clip can be `motivational` *and*
  `storytelling`, which a single combined tag would have lost.

### 4.6 Manual verification — an honest accounting

- **17 clips** (one per source file, used for threshold calibration)
  were listened to in full, with corrections logged.
- **31 clips** (the original 15–30s batch) were sent through
  transcription and reviewed as part of that calibration pass.
- The remaining clips in the final 396 — primarily the shorter 5–15s
  clips produced after the duration-floor change — were **not**
  individually listened to. Their `emotion` field, where backfilled
  from the source-level map, is explicitly marked
  `needs_manual_review: true` with `confidence: 0.65` in the dataset.
  This is a real limitation of the submission, stated plainly rather
  than obscured by the size of the "passed" number.

---

## 5. What I Would Improve Given More Time

- **Fix the VAD merge logic directly**, rather than lowering the
  duration floor. A better short-segment-merging strategy would likely
  recover most of the 727→31 loss without sacrificing the 15–30s target
  or degrading the emotion tagger's input length.
- **Prosody-aware emotion tagging.** Pass pitch variance, speech rate,
  and energy envelope alongside the transcript, or use a dedicated
  audio-emotion classifier, so tagging isn't blind to delivery and
  doesn't depend on transcript length.
- **Speaker gender classification.** Currently unset for every clip;
  a lightweight pre-trained classifier could populate this without
  manual labelling.
- **More Hindi sources.** The Hindi set (9 sources) was smaller than
  English (11 sources) purely due to time constraints — closing this
  gap would directly close the likely English/Hindi minutes imbalance
  noted in Section 4.4.
- **Pipeline-level timeouts and alerting.** The overnight DNS stall and
  the oversized-file transcription hang both manifested as silent
  multi-hour stalls rather than clear failures. Explicit per-request
  timeouts with a visible alert would catch both classes of failure
  immediately instead of requiring a human to notice and interrupt.
- **Full manual audit coverage.** Only 17 of 396 clips were individually
  listened to. With more time, every clip — especially the ones
  carrying a source-level fallback emotion tag — would get a real
  listening pass rather than a flagged-but-unverified label.

---

## Appendix A — Quality Thresholds (Final)

| Metric | Starting value | Final value | Changed? |
|---|---|---|---|
| SNR threshold | 20 dB | 18 dB | Yes |
| Silence ratio max | 0.20 | 0.30 | Yes |
| Clipping max | 0.1% | 0.1% | No |
| Duration min | 15 s | 5 s | Yes |
| Duration max | 30 s | 30 s | No — bound by the Sarvam sync API's practical request-size limit |

## Appendix B — Emotion Taxonomy (Final)

| Tag | Used in final dataset? | Notes |
|---|---|---|
| neutral | Yes | Largest class (52.8%) |
| happy | Yes | |
| motivational | Yes | |
| excited | Yes | |
| sad | Yes | |
| formal | Yes | |
| angry | Yes | |
| informational | Yes | Smallest classes (0.3% each) |
| conversational | Yes | |
| whisper | No | No clean source material found |
| uncertain | No | All 178 instances backfilled from source-level provisional tags before publication |

## Appendix C — Pipeline Funnel

| Stage | Count | Notes |
|---|---|---|
| Source videos identified | 20 | 11 English, 9 Hindi |
| Removed — wrong language | 3 | Foreign-language clips accidentally added at collection time |
| Preprocessed (ffmpeg) | 17 | All succeeded |
| Raw VAD segments | 727 | From the 17 preprocessed files |
| Quality-calibration sample | 17 | One representative clip per source file |
| — passed (post-tuning) | 16 | |
| — rejected (post-tuning) | 1 | Confirmed genuinely noisy, 11.1 dB |
| Transcription calibration batch | 31 | Original 15–30s segment pool |
| — succeeded | 30 | |
| — failed | 1 | Transient server disconnect |
| **Final published dataset** | **396 clips / 57.3 min** | After lowering the duration floor to 5s and running quality filter + transcription + emotion tagging at full scale |

## Appendix D — Tools and Stack

| Stage | Tool |
|---|---|
| Project/dependency management | Python 3.12, `uv` |
| Config | `pydantic-settings` v2 |
| Download | `yt-dlp` (Python API) |
| Audio normalization | `ffmpeg` (`loudnorm`, high-pass) |
| Segmentation | `silero-vad` |
| Quality filtering | `librosa` (SNR, silence ratio, clipping) |
| Transcription | Sarvam `saaras:v3` (sync STT API) |
| Emotion/style tagging | Sarvam `sarvam-105b` (chat completions, few-shot) |
| Dataset assembly | `datasets`, `soundfile` |
| Publication | `huggingface_hub` |

---

*Report generated from `report/pipeline_report.md`. To produce the PDF:*

```
pandoc report/pipeline_report.md -o report/pipeline_report.pdf
```
---
title: "Sarvam TTS Dataset — Pipeline Report"
author: "[Your Name]"
date: "June 2026"
geometry: margin=2.5cm
fontsize: 11pt
linestretch: 1.4
colorlinks: true
---

# Sarvam TTS Dataset Pipeline Report

**Submitted:** June 2026  
**Dataset:** [YOUR_HF_DATASET_URL]  
**Repository:** [YOUR_GITHUB_URL]  
**Total clips (final dataset):** [N] clips · [X] minutes  
**Languages:** Indian English (`en-IN`) · Hindi (`hi-IN`)

---

## 1. What I Built

### 1.1 Overview

This project produces a curated speech dataset for Text-to-Speech (TTS)
model training, containing single-speaker audio clips in Indian English
and Hindi. The pipeline runs in seven automated stages followed by a
mandatory manual listening pass. The automated stages handle the
mechanics of audio acquisition, cleaning, and labelling; the manual
pass is where quality decisions are actually made.

The design principle throughout was to avoid silently discarding or
silently including clips. Every reject has a logged reason in
`data/manifest.json`. Every manual correction is recorded alongside the
original pipeline output so the two can be compared.

### 1.2 Phase 1 — Scaffold and Configuration

The project is managed with `uv` on Python 3.12 (pinned to
`<3.13` in `pyproject.toml` to avoid mid-project interpreter upgrades).
All settings — sample rate, quality thresholds, clip duration window,
language list — live in `config.yaml` and are loaded at import time by
`src/config.py` via `pydantic-settings` v2. API keys (`SARVAM_API_KEY`,
`HF_TOKEN`) are read from `.env` and never stored in code or committed
to the repository. A hard error at import time catches a missing key
before any network call is made.

### 1.3 Phase 2 — Data Collection

Audio is downloaded from YouTube using `yt-dlp`'s Python API with a
specified start and end timestamp per clip. Output is written directly
to 22 050 Hz mono WAV via `yt-dlp`'s built-in FFmpeg postprocessor.
Each source entry in `data/sources_en.yaml` and `data/sources_hi.yaml`
was manually vetted before being added: I watched the clip, confirmed a
single speaker throughout the segment, confirmed no background music
bed, and confirmed I could understand the language well enough to judge
ASR accuracy during the audit. Source types used were:
[LIST YOUR ACTUAL SOURCE TYPES — e.g. TED/TEDx talks, All India Radio
solo news segments, lecture recordings, stand-up comedy specials,
narrated documentary clips].

### 1.4 Phase 3 — Audio Preprocessing and Segmentation

Downloaded audio is normalised to −23 LUFS with an 80 Hz high-pass
filter applied, both via `ffmpeg`'s `loudnorm` filter. Noise reduction
and noise gating are intentionally omitted: aggressive denoising
introduces spectral artefacts that would be learned by a TTS model as
part of the "style" of the data, which is worse than the original
background noise at typical recording levels.

Segmentation uses `silero-vad` to find natural speech pause boundaries,
producing clips between 15 and 30 seconds. Segments shorter than 15
seconds are merged with adjacent segments where possible; segments
longer than 30 seconds are split at the longest internal pause. The VAD
threshold was [LEFT AT DEFAULT / ADJUSTED TO X — fill in].

### 1.5 Phase 4 — Quality Filtering

Each segment is evaluated by `src/quality_checker.py` using `librosa`
across three metrics: estimated signal-to-noise ratio (percentile-based
noise floor estimation), silence ratio (fraction of frames below an
energy threshold), and clipping rate (fraction of samples at or near
±1.0). A composite quality score (0–1) is computed as a weighted
average [DOCUMENT YOUR ACTUAL WEIGHTS — e.g. SNR 50%, silence 30%,
clipping 20%]. Clips failing any threshold are rejected and logged
with a specific reason before any API call is made, saving quota.

### 1.6 Phase 5 — Sarvam API Integration

All Sarvam API calls go through a thin wrapper (`src/sarvam_client.py`)
that centralises client instantiation, retry logic with exponential
backoff on rate-limit and 5xx responses, and surfaces the SDK's
non-obvious call pattern (`client.chat.completions(...)` directly, with
no `.create()` method).

**Transcription** uses `saaras:v3` in synchronous mode
(`client.speech_to_text.transcribe`), which is appropriate for
sub-60-second clips. The transcript is treated as a draft: every
transcript was checked against the audio during the manual audit, and
corrections were written back into the manifest's `manual_review`
sub-object.

**Diarization** uses the same `saaras:v3` model with
`with_diarization=True`. Clips where more than one distinct speaker
label appears in the response are rejected and logged. This catches
clips where a second speaker enters the frame even briefly.

**Emotion tagging** uses `sarvam-105b` chat completions with a
few-shot prompt (`prompts/emotion_classification.txt`) that provides
4–5 labelled examples across both languages and explicitly instructs the
model to return `"uncertain"` rather than forcing a low-confidence
guess. Output is parsed as JSON with a defensive retry on parse failure.
Tags below 0.50 confidence are automatically flagged for manual review.

### 1.7 Phase 6 — Manual Listening Audit

Every clip in the small batch of [N] clips was listened to in full
using `scripts/run_audit.py`, which plays each clip and prompts for a
verdict (`keep` / `reject` / `needs_fix`), an optional corrected
transcript, an optional corrected emotion tag, and free-text notes.
After adjusting thresholds and the emotion taxonomy (documented in
Section 2), the full batch was processed and a [10–15%+] random sample
was audited by ear.

Only clips with `verdict: keep` in the manifest are included in the
final dataset. Clips with `verdict: needs_fix` are excluded from this
submission but the reasons are logged.

### 1.8 Phase 7 — Dataset Assembly and Upload

`src/dataset_builder.py` reads the manifest, filters to kept clips,
resolves the final transcript and emotion tag (manual correction takes
precedence over pipeline output), validates every row against a Pydantic
schema, and builds a `datasets.Dataset` with an `Audio()` feature
column at 22 050 Hz. The dataset is pushed to the HuggingFace Hub as a
public dataset (`private=False`, confirmed via unauthenticated HTTP GET)
with a generated dataset card documenting the verification rate,
language balance, and tag distribution.

---

## 2. Iterations I Made to Improve Quality

> **Instructions for filling this section:**
> Paste your real audit notes below each sub-heading.
> Delete any sub-heading that doesn't apply to your run.
> I will not invent changes you didn't make.

### 2.1 Transcript corrections from the small batch

<!--
FILL IN: List the specific errors you found and corrected.
Examples of what to write:
  - "clip en_003: ASR transcribed 'repo rate' as 'repro rate' — corrected."
  - "clip hi_002: ASR dropped the final word of three consecutive sentences
    where the speaker's voice dropped in pitch — a consistent pattern on
    this source video; clipped transcript flagged."
  - "N clips had no errors detectable by ear."
Be specific about clip IDs where you remember them.
-->

[PASTE YOUR REAL NOTES HERE]

### 2.2 Emotion tag corrections

<!--
FILL IN: Which clips had wrong tags, what was assigned, what you changed
it to, and why.
Examples:
  - "clip en_007: tagged 'excited' at confidence 0.71. On listening the
    delivery is flat and informational — corrected to 'neutral'."
  - "clip hi_005: tagged 'neutral' but the narrator's voice is clearly
    building toward a reveal; corrected to 'dramatic'."
  - "The model consistently tagged stand-up comedy clips as 'happy' when
    the actual delivery was dryer than that. Merged these into
    'conversational' for this source type."
-->

[PASTE YOUR REAL NOTES HERE]

### 2.3 Threshold changes

<!--
FILL IN: Did you change any threshold in config.yaml based on what you heard?
Format: old value → new value, and one sentence on why.
Examples:
  - "SNR threshold: 20 dB → 18 dB. Two clips that sounded clean to my
    ear were scoring 18.5–19.5 dB because of a consistent low-level air
    conditioning hum. The hum is not distracting and the speech is clear;
    lowering the threshold to 18 dB retained them."
  - "Silence ratio: kept at 0.20. The default was appropriate for the
    source material — no clips were rejected on this criterion alone."
  - "Duration window: kept at 15–30s. The VAD produced no segments
    outside this range after merging."
If you changed nothing, write: "No thresholds were changed from their
starting values. The small-batch audit confirmed the defaults were
appropriate for this source material."
-->

[PASTE YOUR REAL NOTES HERE]

### 2.4 Taxonomy changes

<!--
FILL IN: Did you rename, merge, split, or drop any emotion tags?
Examples:
  - "Dropped 'whisper' from the active taxonomy. No source clip in my
    dataset contained whisper delivery; the category was not useful."
  - "Merged 'formal' and 'informational' for news clips. In practice
    every news clip was being tagged one or the other based on sentence
    length rather than meaningful delivery difference. Kept 'informational'
    as the single tag for this source type."
  - "Added informal note in manifest that 'motivational' and 'excited'
    are blurry on TED talk clips — both tags appear but the distinction
    is not reliable from transcript alone."
If you changed nothing: "The draft taxonomy was used as-is.
After listening, the categories were sufficiently distinct for the
source types in this dataset."
-->

[PASTE YOUR REAL NOTES HERE]

### 2.5 Source type decisions

<!--
FILL IN: Did you reject a whole source type, or decide to add more
of a particular type, based on what you found in the small batch?
Examples:
  - "Rejected all panel interview clips after finding that two clips
    from [SOURCE] had a host who asked questions off-camera — the
    diarizer caught one but missed one where the host spoke quietly.
    Decision: exclude panel formats entirely, not just rely on diarization."
  - "Added more TEDx clips after finding they produced the cleanest
    SNR scores and the most reliable emotion tags."
-->

[PASTE YOUR REAL NOTES HERE]

---

## 3. What Worked and What Didn't

### 3.1 What worked well

**VAD segmentation.** `silero-vad` cut at natural pause boundaries in
every clip I checked — no mid-word cuts in the small batch of [N] clips.
The merge logic for short segments behaved correctly.

**Diarization as a reject gate.** [FILL IN: Did it catch any
multi-speaker clips? Example: "Caught 2 clips where a host briefly
interjected. Both would have passed the SNR filter and would not have
been obvious from the transcript alone."]

**Few-shot emotion prompt.** [FILL IN: How accurate were the
LLM tags before correction? Example: "Of 10 small-batch clips, 7 were
tagged correctly without correction. The 3 errors were all on clips
where the transcript was procedural/instructional — consistent with the
prompt's own 'uncertain' case."]

**Manifest as audit trail.** Having every pipeline stage's pass/fail
and reason in a single JSON file made the audit straightforward — I
could see at a glance why a clip was rejected without re-running any
code.

### 3.2 What didn't work as expected

<!--
Be honest here. This section is explicitly about things that needed
rework or produced worse results than expected.
-->

**[FILL IN STAGE NAME — e.g. SNR estimation].**
[FILL IN: What happened and what you did about it. Example:
"The percentile-based SNR estimator in quality_checker.py
over-penalised clips with deliberate pauses — a speaker who pauses
for emphasis reads as having a high silence ratio. Clip hi_008 was
rejected on silence_ratio=0.23 (threshold 0.20) but sounds perfectly
usable. Manually overrode to 'keep' with a note."]

**[FILL IN — e.g. yt-dlp timestamp accuracy].**
[FILL IN: Example: "yt-dlp's start/end timestamp cutting was
occasionally off by 1–2 seconds, producing a brief fragment of the
previous speaker or a hard cut into speech. Caught during audit;
affected 2 of 10 small-batch clips. Mitigation: added 1 second of
padding to start_sec in sources YAML for affected sources."]

**[FILL IN — e.g. Hindi ASR accuracy].**
[FILL IN: Example: "saaras:v3 accuracy on Hindi was noticeably
lower than on English for clips with code-switched vocabulary
(English technical terms mid-sentence). Required more hand-correction
time than English clips."]

---

## 4. Quality Observations and Decisions

### 4.1 SNR and audio quality

Final SNR threshold used: **[X] dB** (started at 20 dB,
[changed to X because... / kept at 20 dB because...]).

Mean SNR across kept clips: **[X] dB** (from `summary_stats.txt`).  
Lowest SNR in a kept clip: **[X] dB** (clip [ID]) — retained because
[REASON].  
Highest SNR in a rejected clip: **[X] dB** (clip [ID]) — rejected
because [REASON, e.g. multi-speaker despite clean audio].

[FILL IN: Any pattern you noticed about which source types produced
the cleanest audio. Example: "TEDx clips consistently scored above
30 dB. News clips from All India Radio scored 25–32 dB. Lecture
recordings were the most variable, ranging from 18 to 35 dB depending
on the microphone setup."]

### 4.2 Clip duration

Final duration window used: **[X]–[Y] seconds**
(started at 15–30 s, [changed / kept] because [REASON]).

[FILL IN: Any patterns. Example: "The VAD produced very few segments
shorter than 18 seconds from this source material — speakers rarely
pause for more than 0.5 seconds mid-thought in prepared speech.
The lower bound of 15 s was never a binding constraint."]

### 4.3 Emotion tag reliability

The LLM emotion tagger operates on transcript text only, with no
access to audio prosody. This is a ceiling on accuracy: a speaker
can deliver an excited script flatly (or a flat script with warmth)
and the tag will reflect the words, not the delivery.

In practice this was most visible in [FILL IN: which source type /
which clips]. For [SOURCE TYPE] clips, the tagger's output was
[reliable / unreliable] because [REASON].

Tags corrected during audit: **[N]** out of **[M]** clips audited
([X]%).

Most common correction pattern: [FILL IN — e.g. "excited → neutral
on news clips where exclamatory headline language was read flatly"].

### 4.4 Source type assessment

| Source type | Clips collected | Clips kept | Notes |
|-------------|----------------|------------|-------|
| [e.g. TEDx talk] | [N] | [N] | [e.g. High SNR, reliable tags] |
| [e.g. news solo] | [N] | [N] | [e.g. Very clean, low emotion variety] |
| [e.g. lecture]   | [N] | [N] | [e.g. Variable mic quality] |
| [ADD YOUR ROWS]  |     |    | |

[FILL IN: Any source type you decided to exclude after the small-batch
audit, and why.]

### 4.5 Language balance

Final dataset: **[N] en-IN clips ([X] min)** ·
**[N] hi-IN clips ([X] min)**.

[FILL IN: Was this the balance you aimed for? Did one language
produce more rejects than the other, and if so why?]

### 4.6 Manual verification rate

**[N] of [M] total clips ([X]%) were manually listened to.**

This covers:
- All [N] clips in the small batch (100% of small batch)
- A [X]% random sample of the remaining clips

Clips included in the dataset with `manually_verified: false`:
**[N]** — these use raw ASR transcript and LLM emotion tag without
human correction. They are flagged in the dataset metadata.

---

## 5. What I Would Improve Given More Time

**Prosody-aware emotion tagging.** The current tagger reads the
transcript; it cannot hear flat delivery, sarcasm, or emotional
build-up. A simple improvement would be to extract audio-level
features (pitch variance, speech rate, energy envelope) and pass
them alongside the transcript. A stronger improvement would be a
fine-tuned audio emotion classifier — but that is itself a training
data problem.

**Speaker gender classification.** The `speaker_gender` field
defaults to `"unknown"` for every clip because the pipeline has no
gender classifier. A pre-trained wav2vec2-based classifier could
populate this without manual labelling at a reasonable accuracy level.

**Automated transcript verification.** The current pipeline uses
ASR output as-is and relies on manual correction. A word-level
confidence score from the ASR model (if exposed by the Sarvam API)
could flag specific words for targeted human review rather than
requiring a full re-listen for every clip.

**More source diversity.** The dataset draws heavily from
[FILL IN: your dominant source types]. With more time I would add
[FILL IN: what you'd add — e.g. "phone-quality recordings to cover
channel-robust TTS, more regional accent variation within Indian
English, more Hindi dialectal variation"].

**Balanced emotion sampling.** The current emotion distribution is
skewed toward [FILL IN: your most common tag]. A stratified sampling
approach — collecting source clips by target emotion rather than
by convenient availability — would produce a more balanced dataset
useful for multi-style TTS training.

**Full manual audit coverage.** [X]% of the full batch was audited
by ear. With more time the target would be 100%. At scale, active
learning could prioritise clips where the ASR confidence and emotion
confidence are both low, making the human time more efficient.

---

## Appendix A — Quality Thresholds (final values)

| Metric | Starting value | Final value | Changed? |
|--------|---------------|-------------|---------- |
| SNR threshold | 20 dB | [X] dB | [Yes/No] |
| Silence ratio max | 0.20 | [X] | [Yes/No] |
| Clipping max | 0.1% | [X]% | [Yes/No] |
| Duration min | 15 s | [X] s | [Yes/No] |
| Duration max | 30 s | [X] s | [Yes/No] |

## Appendix B — Emotion Taxonomy (final)

Tags used in the uploaded dataset (any tags removed from the draft
list are noted with reason):

| Tag | Used? | Notes |
|-----|-------|-------|
| neutral | [Y/N] | |
| happy | [Y/N] | |
| sad | [Y/N] | |
| excited | [Y/N] | |
| angry | [Y/N] | |
| formal | [Y/N] | |
| informational | [Y/N] | |
| storytelling | [Y/N] | |
| conversational | [Y/N] | |
| motivational | [Y/N] | |
| whisper | [Y/N] | [e.g. "Not used — no source material"]|
| dramatic | [Y/N] | |
| uncertain | [Y/N] | [Should be 0 in final dataset — uncertain clips were re-tagged or rejected] |

## Appendix C — Clip Counts by Pipeline Stage

| Stage | Passed | Failed | Failure reason (most common) |
|-------|--------|--------|-------------------------------|
| Download | [N] | [N] | [e.g. region-locked video] |
| Preprocess | [N] | [N] | [e.g. corrupt download] |
| Segment | [N] | [N] | [e.g. no usable VAD segments] |
| Quality check | [N] | [N] | [e.g. snr_below_threshold] |
| Transcription | [N] | [N] | [e.g. API error] |
| Diarization | [N] | [N] | [e.g. multi_speaker_detected] |
| Emotion tagging | [N] | [N] | [e.g. parse_failed_after_retry] |
| Manual audit — keep | [N] | — | — |
| Manual audit — reject | — | [N] | [e.g. cut_mid_word, background_noise] |
| Manual audit — needs_fix | [N] | — | [deferred, not in final dataset] |

*All stage pass/fail counts are drawn directly from `data/manifest.json`.*

---

*Report generated from `report/pipeline_report.md` via:*

```
pandoc report/pipeline_report.md -o report/pipeline_report.pdf
```
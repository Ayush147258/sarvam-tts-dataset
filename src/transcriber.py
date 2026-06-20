"""
Transcribes a single preprocessed audio clip using Sarvam's saaras:v3
speech-to-text model via the synchronous REST API.

SYNC vs BATCH — decision documented here per the build guide instruction:
The Sarvam docs explicitly categorise these as two separate APIs:
  - REST API (synchronous) : files under 30 seconds — immediate response
  - Batch API (async)      : files up to 2 hours — submit → poll → download
Our clips are 15-30 seconds (segmenter target from config.yaml), which
puts them squarely in the synchronous category. The batch job-submit ->
poll -> download-result flow is intentionally NOT used here:
  1. It adds polling latency (seconds to minutes) for clips that get an
     immediate response synchronously.
  2. More moving parts = more failure modes for no benefit on short audio.
  3. Sarvam's own docs say "For short audio files (<30 seconds), you can
     skip this step and directly proceed with transcription using the
     real-time API." (docs.sarvam.ai cookbook, stt-translate-api-tutorial)
The clip duration limit from config.yaml is 30 seconds. If you later
change max_sec above 30, you WILL need to switch to the batch API here.

CALL PATTERN -- verified against installed sarvamai==0.1.28 source:
    client.speech_to_text.transcribe(
        file=<open file object>,
        model="saaras:v3",
        mode="transcribe",
        language_code="en-IN",   # or "hi-IN"
    )
No .create() -- that method does not exist on this SDK. See the SKILL.md
in Sarvam's own `sarvamai/skills` GitHub repo for a full list of SDK
gotchas documented for AI coding agents.

RESPONSE SHAPE -- verified against SpeechToTextResponse Pydantic model:
    response.request_id           str
    response.transcript           str   <- the main output
    response.timestamps           list | None
    response.diarized_transcript  DiarizedTranscript | None
    response.language_code        str | None
    response.language_probability float | None

Usage:
    from pathlib import Path
    from src.transcriber import transcribe_clip

    result = transcribe_clip(Path("processed/en_001_seg0.wav"), "en-IN")
    print(result["transcript"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from rich.console import Console
from sarvamai.core.api_error import ApiError

from src.sarvam_client import call_with_retry, get_client

console = Console()

SAARAS_MODEL = "saaras:v3"
TRANSCRIBE_MODE = "transcribe"

SYNC_API_MAX_SEC_RECOMMENDED = 30


def _get_audio_duration_sec(audio_path: Path) -> Optional[float]:
    """
    Return clip duration in seconds using soundfile. Returns None if the
    file can't be read -- caller treats this as a warning, not a hard stop.
    """
    try:
        import soundfile as sf
        info = sf.info(str(audio_path))
        return info.duration
    except Exception:
        return None


def transcribe_clip(audio_path: Path, language_code: str) -> Dict:
    """
    Transcribe a single audio clip using Sarvam saaras:v3 (synchronous API).

    Args:
        audio_path    : Path to a preprocessed 22050 Hz mono WAV file.
        language_code : BCP-47 language tag -- "en-IN" or "hi-IN".

    Returns:
        {
            "transcript"          : str,
            "language_code"       : str | None,
            "language_probability": float | None,
            "raw_response"        : dict,
            "success"             : bool,
            "error"               : str | None,
        }
    """
    def _FAILURE(reason):
        return {
            "transcript": "",
            "language_code": None,
            "language_probability": None,
            "raw_response": {},
            "success": False,
            "error": reason,
        }

    if not audio_path.exists():
        msg = f"Audio file not found: {audio_path}"
        console.log(f"[red]X[/red] {msg}")
        return _FAILURE(msg)

    duration = _get_audio_duration_sec(audio_path)
    if duration is not None and duration > SYNC_API_MAX_SEC_RECOMMENDED:
        console.log(
            f"[yellow]![/yellow] {audio_path.name} is {duration:.1f}s, "
            f"exceeding the recommended sync API limit of "
            f"{SYNC_API_MAX_SEC_RECOMMENDED}s. Consider the Batch API "
            f"for files longer than 30s. Proceeding anyway."
        )

    client = get_client()
    if duration:
        console.log(
            f"[cyan]->[/cyan] Transcribing [white]{audio_path.name}[/white] "
            f"({duration:.1f}s, {language_code}, model={SAARAS_MODEL})"
        )
    else:
        console.log(
            f"[cyan]->[/cyan] Transcribing [white]{audio_path.name}[/white] "
            f"({language_code}, model={SAARAS_MODEL})"
        )

    try:
        with open(audio_path, "rb") as f:
            response = call_with_retry(
                lambda: client.speech_to_text.transcribe(
                    file=(audio_path.name, f, "audio/wav"),
                    model=SAARAS_MODEL,
                    mode=TRANSCRIBE_MODE,
                    language_code=language_code,
                ),
                label=f"transcribe:{audio_path.name}",
            )

    except ApiError as e:
        msg = f"API error transcribing {audio_path.name}: {e}"
        console.log(f"[red]X[/red] {msg}")
        return _FAILURE(msg)
    except Exception as e:
        msg = f"Unexpected error transcribing {audio_path.name}: {e}"
        console.log(f"[red]X[/red] {msg}")
        return _FAILURE(msg)

    transcript = response.transcript or ""

    if not transcript.strip():
        console.log(
            f"[yellow]![/yellow] {audio_path.name}: API returned empty "
            f"transcript. The clip may be silent, wrong language specified, "
            f"or audio quality too low for the model."
        )

    preview = transcript[:80] + ('...' if len(transcript) > 80 else '')
    console.log(f"[green]OK[/green] {audio_path.name}: \"{preview}\"")

    try:
        raw = response.model_dump()
    except Exception:
        raw = {"transcript": transcript}

    return {
        "transcript": transcript,
        "language_code": response.language_code,
        "language_probability": response.language_probability,
        "raw_response": raw,
        "success": True,
        "error": None,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        console.log(
            "[yellow]Usage:[/yellow] uv run python -m src.transcriber "
            "<wav_path> <language_code>\n"
            "  language_code: en-IN or hi-IN"
        )
        sys.exit(1)

    path = Path(sys.argv[1])
    lang = sys.argv[2]

    result = transcribe_clip(path, lang)

    if result["success"]:
        console.log(f"[bold green]Success.[/bold green]")
        console.log(f"Transcript     : {result['transcript']}")
        console.log(f"Language code  : {result['language_code']}")
        console.log(f"Lang prob      : {result['language_probability']}")
        console.log(f"Raw keys       : {list(result['raw_response'].keys())}")
    else:
        console.log(f"[bold red]Failed:[/bold red] {result['error']}")
        sys.exit(1)
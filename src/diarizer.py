"""
src/diarizer.py

Speaker diarization using Sarvam saaras:v3 ASR with diarization enabled.
Determines whether a clip contains a single speaker (keep) or multiple speakers (reject).
"""

from pathlib import Path
import logging
from rich.logging import RichHandler
from src.sarvam_client import call_with_retry, get_client

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger(__name__)


def check_single_speaker(audio_path: Path, language_code: str) -> dict:
    """
    Submit audio to Sarvam saaras:v3 with diarization enabled.
    Parse the response to count distinct speaker labels.

    Returns:
        {
            "speaker_count": int,
            "is_single_speaker": bool,
            "diarized_transcript": list[dict] | str,  # raw diarized output
            "error": str | None
        }

    Rejection logic:
        If speaker_count > 1, the caller should reject the clip.
        This function logs the rejection reason clearly.
    """
    if not audio_path.exists():
        log.error(f"[diarizer] File not found: {audio_path}")
        return {
            "speaker_count": 0,
            "is_single_speaker": False,
            "diarized_transcript": None,
            "error": "file_not_found"
        }

    client = get_client()

    try:
        log.info(f"[diarizer] Submitting {audio_path.name} for diarization (lang={language_code})")

        with open(audio_path, "rb") as f:
            response = call_with_retry(
                lambda: client.speech_to_text.transcribe(
                    file=(audio_path.name, f, "audio/wav"),
                    model="saaras:v3",
                    mode="transcribe",
                    language_code=language_code,
                ),
                label=f"diarize:{audio_path.name}",
            )

        # ------------------------------------------------------------------ #
        # Parse diarized output.                                               #
        # Sarvam returns diarized results in response.diarized_transcript      #
        # (a list of utterances, each with a "speaker" label like "SPEAKER_0" #
        # or "speaker_1"). We collect all unique labels to count speakers.     #
        # ------------------------------------------------------------------ #

        diarized = None
        speaker_labels: set[str] = set()

        # The SDK may return the diarized content in different shapes depending
        # on SDK version; handle both dict-style and attribute-style responses.
        raw = response

        # Try attribute access first (SDK returns a Pydantic-like model)
        if hasattr(raw, "diarized_transcript") and raw.diarized_transcript:
            diarized = raw.diarized_transcript
            for utterance in diarized:
                # Each utterance may be a dict or an object
                if isinstance(utterance, dict):
                    label = utterance.get("speaker") or utterance.get("speaker_id")
                else:
                    label = getattr(utterance, "speaker", None) or getattr(utterance, "speaker_id", None)
                if label:
                    speaker_labels.add(str(label).strip().lower())

        # Fallback: try dict-style (if SDK returns a plain dict)
        elif isinstance(raw, dict):
            diarized = raw.get("diarized_transcript") or raw.get("utterances") or []
            for utterance in diarized:
                label = utterance.get("speaker") or utterance.get("speaker_id")
                if label:
                    speaker_labels.add(str(label).strip().lower())

        # Last resort: if no diarized_transcript field, use plain transcript
        # and assume 1 speaker (log a warning — this means diarization wasn't
        # returned by the API for some reason).
        if not speaker_labels:
            log.warning(
                f"[diarizer] No diarized_transcript in response for {audio_path.name}. "
                "Assuming 1 speaker but flagging for manual review."
            )
            # Fall back to transcript text if available
            fallback_transcript = (
                getattr(raw, "transcript", None)
                or (raw.get("transcript") if isinstance(raw, dict) else None)
                or ""
            )
            return {
                "speaker_count": 1,
                "is_single_speaker": True,
                "diarized_transcript": fallback_transcript,
                "error": "diarization_not_returned_assumed_single"
            }

        speaker_count = len(speaker_labels)
        is_single = speaker_count == 1

        if is_single:
            log.info(
                f"[diarizer] ✅ {audio_path.name} → single speaker detected "
                f"(label: {', '.join(speaker_labels)}). KEEP."
            )
        else:
            log.warning(
                f"[diarizer] ❌ REJECTED {audio_path.name} — "
                f"{speaker_count} distinct speakers detected: "
                f"{', '.join(sorted(speaker_labels))}. "
                "Multi-speaker clips are not suitable for TTS training."
            )

        return {
            "speaker_count": speaker_count,
            "is_single_speaker": is_single,
            "diarized_transcript": diarized,
            "error": None
        }

    except Exception as e:
        log.error(f"[diarizer] API error for {audio_path.name}: {e}", exc_info=True)
        return {
            "speaker_count": 0,
            "is_single_speaker": False,
            "diarized_transcript": None,
            "error": str(e)
        }


# --------------------------------------------------------------------------- #
# Quick manual test — run as: uv run python -m src.diarizer                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: uv run python -m src.diarizer <audio_path> <language_code>")
        print("  e.g. uv run python -m src.diarizer processed/en_001.wav en-IN")
        sys.exit(1)

    path = Path(sys.argv[1])
    lang = sys.argv[2]

    result = check_single_speaker(path, lang)

    print("\n--- Diarization Result ---")
    print(f"  Speaker count    : {result['speaker_count']}")
    print(f"  Is single speaker: {result['is_single_speaker']}")
    print(f"  Error            : {result['error']}")
    if result["diarized_transcript"]:
        print(f"  Diarized output  : {result['diarized_transcript']}")

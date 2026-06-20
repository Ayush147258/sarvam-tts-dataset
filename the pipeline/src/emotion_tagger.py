"""
src/emotion_tagger.py

Emotion and style classification for TTS dataset clips using sarvam-105b.

Loads the few-shot prompt template from prompts/emotion_classification.txt,
fills in the transcript + metadata, calls sarvam-105b chat completions,
and parses the JSON response defensively.

Call pattern: client.chat.completions(...) — NO .create() method on this SDK.
"""

from pathlib import Path
import json
import logging
import re
from typing import Sequence, Any

from rich.logging import RichHandler

from src.config import PIPELINE_ROOT
from src.sarvam_client import get_client

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE_PATH = PIPELINE_ROOT / "prompts" / "emotion_classification.txt"

VALID_EMOTION_TAGS = {
    "neutral", "happy", "sad", "excited", "angry", "formal",
    "informational", "storytelling", "conversational", "motivational",
    "whisper", "dramatic", "uncertain",
    # The few-shot example uses "motivated" — accept it as an alias
    "motivated",
}

VALID_STYLE_TAGS = {
    "neutral", "happy", "sad", "excited", "angry", "formal",
    "informational", "storytelling", "conversational", "motivational",
    "whisper", "dramatic",
}

MODEL = "sarvam-105b"
FALLBACK_MODEL = "sarvam-30b"   # used if 105b is rate-limited

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_template() -> str:
    """Load the prompt template once; crash loudly if it's missing."""
    if not PROMPT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Prompt template not found at {PROMPT_TEMPLATE_PATH}. "
            "Run from the project root and ensure prompts/ directory exists."
        )
    return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")


def _fill_template(template: str, transcript: str, language: str, source_type: str) -> str:
    """
    Substitute the three placeholders in the template.
    Using str.replace() rather than .format() to avoid KeyError on the
    many curly braces used for the JSON examples in the prompt body.
    """
    filled = template.replace("{transcript}", transcript.strip())
    filled = filled.replace("{language}", language.strip())
    filled = filled.replace("{source_type}", source_type.strip())
    return filled


def _strip_fences(text: str) -> str:
    """
    Remove markdown code fences that some models add despite instructions.
    Handles:
````json ... ```
``` ... ```
        `{...}`
    Returns the innermost content.
    """
    text = text.strip()

    # Match ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        return fence_match.group(1).strip()

    # Match single backtick wrapping: `{...}`
    backtick_match = re.match(r"^`(.+)`$", text, re.DOTALL)
    if backtick_match:
        return backtick_match.group(1).strip()

    return text


def _extract_json_object(text: str) -> str:
    """
    Last-resort: find the first {...} blob in the text even if there's
    surrounding prose (e.g. "Sure! Here is the JSON: {...} Hope that helps!")
    """
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def _parse_response(raw_text: str) -> dict | None:
    """
    Try to parse the model's response into a valid classification dict.
    Returns None if parsing fails completely.

    Normalises the emotion tag:
      - "motivated" → "motivational" (the few-shot example leaked an alias)
      - lowercases everything
      - strips whitespace
    """
    cleaned = _strip_fences(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try extracting just the JSON object from surrounding prose
        extracted = _extract_json_object(cleaned)
        try:
            data = json.loads(extracted)
        except json.JSONDecodeError:
            return None

    # Validate required keys
    if not all(k in data for k in ("emotion", "style", "confidence")):
        log.warning(f"[emotion_tagger] Response missing required keys: {data}")
        return None

    # Normalise values
    emotion = str(data["emotion"]).strip().lower()
    style   = str(data["style"]).strip().lower()

    # Alias fix: "motivated" is not in VALID_EMOTION_TAGS, remap it
    if emotion == "motivated":
        emotion = "motivational"

    # Confidence: clamp to [0.0, 1.0] and coerce to float
    try:
        confidence = float(data["confidence"])
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.0

    # Warn but don't reject if the tag is outside the taxonomy —
    # flag for manual review instead so we don't silently lose clips
    if emotion not in VALID_EMOTION_TAGS:
        log.warning(
            f"[emotion_tagger] Unrecognised emotion tag '{emotion}' — "
            "keeping value, setting needs_manual_review=True."
        )
    if style not in VALID_STYLE_TAGS:
        log.warning(
            f"[emotion_tagger] Unrecognised style tag '{style}' — "
            "keeping value, setting needs_manual_review=True."
        )

    return {
        "emotion": emotion,
        "style": style,
        "confidence": confidence,
    }


def _call_model(messages: Sequence[dict[str, Any]], model: str) -> str | None:
    """
    Call sarvam-105b chat completions.
    Returns raw response text, or None on hard failure.

    Note: the sarvamai SDK uses client.chat.completions(...) directly —
    there is NO .create() method. This is documented in Sarvam's own
    sarvamai/skills SKILL.md for AI coding agents.
    """
    client = get_client()
    try:
        response = client.chat.completions(
            messages=messages,  # type: ignore
            model=model,
        )

        # Response shape: response.choices[0].message.content  (OpenAI-compatible)
        # Handle both attribute-style and dict-style just in case SDK version differs
        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0]
            if hasattr(msg, "message"):
                return msg.message.content
            if isinstance(msg, dict):
                return msg.get("message", {}).get("content", "")

        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")

        log.warning(f"[emotion_tagger] Unexpected response shape: {response}")
        return None

    except Exception as e:
        err_str = str(e).lower()
        # Surface rate-limit errors so the caller can decide to retry/fallback
        if "rate" in err_str or "429" in err_str:
            raise RateLimitError(str(e)) from e
        log.error(f"[emotion_tagger] API call failed: {e}", exc_info=True)
        return None


class RateLimitError(Exception):
    """Raised when the model returns a rate-limit response."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tag_emotion(
    transcript: str,
    language: str,
    source_type: str,
) -> dict:
    """
    Classify the emotion and style of a TTS clip from its transcript.

    Args:
        transcript  : The ASR transcript of the clip.
        language    : BCP-47 language code, e.g. "en-IN" or "hi-IN".
        source_type : Source category, e.g. "tedx_talk", "news_solo_segment".

    Returns:
        {
            "emotion":             str,   # tag or "uncertain"
            "style":               str,   # tag
            "confidence":          float, # 0.0–1.0
            "needs_manual_review": bool,
            "model_used":          str,
            "error":               str | None,
        }
    """
    # ---- Sanity check input ------------------------------------------------
    if not transcript or not transcript.strip():
        log.warning("[emotion_tagger] Empty transcript — skipping API call.")
        return {
            "emotion": "uncertain",
            "style": "neutral",
            "confidence": 0.0,
            "needs_manual_review": True,
            "model_used": None,
            "error": "empty_transcript",
        }

    # ---- Build the prompt --------------------------------------------------
    template = _load_template()
    filled_prompt = _fill_template(template, transcript, language, source_type)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise audio content classifier. "
                "You output ONLY valid JSON and nothing else. "
                "No preamble, no explanation, no markdown."
            ),
        },
        {
            "role": "user",
            "content": filled_prompt,
        },
    ]

    # ---- First attempt: sarvam-105b ----------------------------------------
    model_used = MODEL
    raw_text = None

    try:
        log.info(f"[emotion_tagger] Calling {MODEL} for transcript ({len(transcript)} chars)…")
        raw_text = _call_model(messages, MODEL)

    except RateLimitError as e:
        log.warning(f"[emotion_tagger] Rate limited on {MODEL}, falling back to {FALLBACK_MODEL}. Error: {e}")
        model_used = FALLBACK_MODEL
        try:
            raw_text = _call_model(messages, FALLBACK_MODEL)
        except Exception as e2:
            log.error(f"[emotion_tagger] Fallback model also failed: {e2}")
            return {
                "emotion": "uncertain",
                "style": "neutral",
                "confidence": 0.0,
                "needs_manual_review": True,
                "model_used": FALLBACK_MODEL,
                "error": f"both_models_failed: {e2}",
            }

    # ---- First parse attempt -----------------------------------------------
    if raw_text:
        log.debug(f"[emotion_tagger] Raw response: {raw_text!r}")
        result = _parse_response(raw_text)

        if result:
            needs_review = (
                result["emotion"] not in VALID_EMOTION_TAGS
                or result["style"] not in VALID_STYLE_TAGS
                or result["confidence"] < 0.50
                or result["emotion"] == "uncertain"
            )
            log.info(
                f"[emotion_tagger] ✅ Tagged → emotion={result['emotion']}, "
                f"style={result['style']}, confidence={result['confidence']:.2f}, "
                f"needs_review={needs_review}"
            )
            return {
                **result,
                "needs_manual_review": needs_review,
                "model_used": model_used,
                "error": None,
            }

    # ---- Retry with stricter reminder --------------------------------------
    log.warning(
        "[emotion_tagger] First parse failed — retrying with stricter JSON reminder."
    )

    retry_messages = messages + [
        {
            "role": "assistant",
            "content": raw_text or "",
        },
        {
            "role": "user",
            "content": (
                "Your previous response was not valid JSON. "
                "Return ONLY a JSON object with exactly three keys: "
                '"emotion", "style", "confidence". '
                "No other text. No markdown. Example: "
                '{"emotion": "neutral", "style": "informational", "confidence": 0.85}'
            ),
        },
    ]

    try:
        raw_text_retry = _call_model(retry_messages, model_used)
    except Exception as e:
        log.error(f"[emotion_tagger] Retry API call failed: {e}")
        raw_text_retry = None

    if raw_text_retry:
        log.debug(f"[emotion_tagger] Retry raw response: {raw_text_retry!r}")
        result = _parse_response(raw_text_retry)

        if result:
            log.info(
                f"[emotion_tagger] ✅ Tagged on retry → emotion={result['emotion']}, "
                f"style={result['style']}, confidence={result['confidence']:.2f}"
            )
            return {
                **result,
                "needs_manual_review": True,   # always flag retried responses
                "model_used": model_used,
                "error": "required_json_retry",
            }

    # ---- Give up — flag for manual review ----------------------------------
    log.error(
        "[emotion_tagger] ❌ Both parse attempts failed. "
        "Clip flagged for manual review. Raw responses logged above."
    )
    return {
        "emotion": "uncertain",
        "style": "neutral",
        "confidence": 0.0,
        "needs_manual_review": True,
        "model_used": model_used,
        "error": f"parse_failed_after_retry | raw={raw_text!r} | retry={raw_text_retry!r}",
    }


# ---------------------------------------------------------------------------
# Manual test — run as: uv run python -m src.emotion_tagger
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Four test cases matching the kinds of transcripts Prompt 9 would produce
    test_cases = [
        {
            "transcript": (
                "The repo rate has been kept unchanged at six-point-five percent. "
                "The committee voted four to two, citing elevated food inflation."
            ),
            "language": "en-IN",
            "source_type": "news_solo_segment",
            "expected_emotion": "neutral",
            "expected_style": "informational",
        },
        {
            "transcript": (
                "Yeh jungle kai sadiyon se yahan hai. Iske pedh itne purane hain "
                "ki unke naam tak log bhool gaye hain. Lekin aaj, in pedho ko ek "
                "naya khatraa saamna karna pad raha hai — insaan ka."
            ),
            "language": "hi-IN",
            "source_type": "narrated_documentary",
            "expected_emotion": "neutral",
            "expected_style": "storytelling",
        },
        {
            "transcript": (
                "And that is the moment I realised — every single child deserved a chance. "
                "So I quit my job the next morning and I never looked back."
            ),
            "language": "en-IN",
            "source_type": "tedx_talk",
            "expected_emotion": "motivational",
            "expected_style": "motivational",
        },
        {
            "transcript": "Open your terminal. Type git init. Then git add dot. Good. Now we look at what that created.",
            "language": "en-IN",
            "source_type": "lecture",
            "expected_emotion": "uncertain",
            "expected_style": "informational",
        },
    ]

    if len(sys.argv) > 1:
        # Allow passing a custom transcript on the command line
        custom = " ".join(sys.argv[1:])
        result = tag_emotion(custom, language="en-IN", source_type="unknown")
        print(json.dumps(result, indent=2))
        sys.exit(0)

    print("\n" + "═" * 60)
    print("  EMOTION TAGGER — MANUAL VERIFICATION RUN")
    print("═" * 60)

    all_passed = True
    for i, case in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {case['source_type']} ({case['language']})")
        print(f"  Transcript: {case['transcript'][:80]}…")

        result = tag_emotion(
            transcript=case["transcript"],
            language=case["language"],
            source_type=case["source_type"],
        )

        emotion_match = result["emotion"] == case["expected_emotion"]
        style_match   = result["style"]   == case["expected_style"]

        print(f"  Result   : emotion={result['emotion']}, style={result['style']}, "
              f"confidence={result['confidence']:.2f}, needs_review={result['needs_manual_review']}")
        print(f"  Expected : emotion={case['expected_emotion']}, style={case['expected_style']}")
        print(f"  Match    : emotion={'✅' if emotion_match else '❌'}, "
              f"style={'✅' if style_match else '❌'}")

        if result["error"]:
            print(f"  Error    : {result['error']}")

        if not (emotion_match and style_match):
            all_passed = False

    print("\n" + "═" * 60)
    print(f"  {'ALL TESTS PASSED ✅' if all_passed else 'SOME TESTS FAILED ❌ — review logs above'}")
    print("═" * 60 + "\n")
'''

---

**Verify command:**

```bash
# Run all four built-in test cases
uv run python -m src.emotion_tagger

# Or test against a real transcript from your Prompt 9 output
uv run python -m src.emotion_tagger "Aaj hum baat karenge India ke sabse bade infrastructure project ke baare mein"
```

---

**What to check during your manual verification pass:**

Run it against 3–4 real transcripts from Prompt 9 and compare the tag to what you *hear* in the audio. The text-only classifier will struggle in two specific cases worth noting for your report:

**Prosody gaps** — a transcript like *"That's wonderful, isn't it"* reads as `happy` but could be delivered with flat sarcasm. If the audio contradicts the text-derived tag, mark `needs_manual_review=True` in the manifest yourself and note it. The report should call out that emotion classification from transcript alone is a ceiling on accuracy, and prosody-aware tagging would require a separate audio-feature model.

**Code-switching** — Hindi clips with English technical terms mid-sentence often confuse the confidence score downward. If you see `confidence < 0.50` consistently on Hindi clips that are clearly `informational`, that's worth documenting as a taxonomy calibration note, not a pipeline bug.

The `needs_manual_review` flag will be `True` automatically when confidence is below 0.50, emotion is `uncertain`, or either tag falls outside the valid taxonomy — so your audit script in Prompt 14 can filter on that field to prioritise which clips to listen to first.

'''

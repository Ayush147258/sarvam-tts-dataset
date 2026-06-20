"""
Central settings for the Sarvam TTS dataset pipeline.

Loads two sources and merges them into one flat Settings object:
  1. `.env`        -> secrets: SARVAM_API_KEY, HF_TOKEN
  2. `config.yaml` -> pipeline parameters (nested in the YAML file for
                      readability, but exposed as flat attributes here,
                      e.g. settings.snr_threshold_db, not
                      settings.quality_thresholds.snr_threshold_db)

Usage:
    from src.config import settings
    settings.snr_threshold_db   -> 20
    settings.sarvam_api_key     -> value from .env

If SARVAM_API_KEY is missing, this raises at import time with a clear
message — fail fast and loud rather than letting a later API call fail
with a confusing auth error three pipeline stages in.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = Path(__file__).resolve().parents[0]
CONFIG_YAML_PATH = PROJECT_ROOT / "config.yaml"
ENV_FILE_PATH = PROJECT_ROOT / ".env"
ENV_TEMPLATE_PATH = PROJECT_ROOT / ".env.template"


def _load_config_yaml(path: Path) -> dict:
    """Read config.yaml and flatten its nested sections into one dict.

    The YAML file groups settings under audio / clip / quality_thresholds /
    paths for human readability. We flatten here so the rest of the
    codebase can just do `settings.snr_threshold_db` without caring about
    which section it lived in.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {path}. "
            "Make sure you're running commands from the project root, "
            "or that config.yaml was created in Prompt 1."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    flat: dict = {}

    audio = raw.get("audio", {})
    flat["sample_rate"] = audio.get("sample_rate", 22050)

    clip = raw.get("clip", {})
    flat["target_clip_min_sec"] = clip.get("target_clip_min_sec", 15)
    flat["target_clip_max_sec"] = clip.get("target_clip_max_sec", 30)

    quality = raw.get("quality_thresholds", {})
    flat["snr_threshold_db"] = quality.get("snr_threshold_db", 20)
    flat["silence_ratio_max"] = quality.get("silence_ratio_max", 0.20)
    flat["clipping_max_pct"] = quality.get("clipping_max_pct", 0.1)

    flat["languages"] = raw.get("languages", ["en-IN", "hi-IN"])

    paths = raw.get("paths", {})
    flat["raw_dir"] = paths.get("raw_dir", "raw")
    flat["processed_dir"] = paths.get("processed_dir", "processed")
    flat["output_dir"] = paths.get("output_dir", "data")

    return flat


class Settings(BaseSettings):
    """Flat settings object combining .env secrets and config.yaml params."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- secrets, from .env ---
    sarvam_api_key: str = Field(default="", alias="SARVAM_API_KEY")
    hf_token: str = Field(default="", alias="HF_TOKEN")

    # --- pipeline params, populated from config.yaml after init (see below) ---
    sample_rate: int = 22050
    target_clip_min_sec: int = 15
    target_clip_max_sec: int = 30
    snr_threshold_db: float = 20
    silence_ratio_max: float = 0.20
    clipping_max_pct: float = 0.1
    languages: List[str] = Field(default_factory=lambda: ["en-IN", "hi-IN"])
    raw_dir: str = "raw"
    processed_dir: str = "processed"
    output_dir: str = "data"


def _build_settings() -> Settings:
    """Construct Settings from .env, then overlay config.yaml values."""
    if not ENV_FILE_PATH.exists():
        raise FileNotFoundError(
            ".env file not found.\n"
            f"Expected it at: {ENV_FILE_PATH}\n"
            f"Fix: copy {ENV_TEMPLATE_PATH.name} to .env and fill in your "
            "real SARVAM_API_KEY and HF_TOKEN values, e.g.:\n"
            "    cp .env.template .env   (or, on Windows PowerShell: "
            "Copy-Item .env.template .env)\n"
            "then edit .env with a text editor."
        )

    yaml_values = _load_config_yaml(CONFIG_YAML_PATH)

    # pydantic-settings reads .env automatically via model_config; we then
    # pass the yaml-derived values as explicit overrides on top.
    settings_instance = Settings(**yaml_values)

    return settings_instance


def require_sarvam_api_key() -> str:
    """Return SARVAM_API_KEY or raise a clear setup error."""
    if not settings.sarvam_api_key or settings.sarvam_api_key == "your_sarvam_api_key_here":
        raise ValueError(
            "SARVAM_API_KEY is missing or still set to the placeholder value.\n"
            f"Fix: open {ENV_FILE_PATH} and set SARVAM_API_KEY to your real key "
            "from https://dashboard.sarvam.ai/ (API Keys section)."
        )
    return settings.sarvam_api_key


def require_hf_token() -> str:
    """Return HF_TOKEN or raise a clear setup error."""
    if not settings.hf_token or settings.hf_token == "your_huggingface_write_token_here":
        raise ValueError(
            "HF_TOKEN is missing or still set to the placeholder value.\n"
            f"Fix: open {ENV_FILE_PATH} and set HF_TOKEN to a HuggingFace token "
            "with write access."
        )
    return settings.hf_token


# Single shared instance, imported everywhere else as `from src.config import settings`
settings = _build_settings()

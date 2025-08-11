# config.py (Python 3.9-safe)
from pathlib import Path
from typing import Union
import os
import yaml

_DEFAULT = {
    "app": {"diarization": True, "samplerate": 16000, "duration": 7},
    "llm": {"model": "gpt-3.5-turbo", "base_system_prompt": "You are a helpful speech agent."},
    "tts": {"voice_id": "OYTbf65OHHFELVut7v2H", "model_id": "eleven_flash_v2"},
    "asr": {"whisper_model": "small"},
    "runtime": {"device": "auto"},
}

def load_config(path: Union[str, Path] = "config.yaml") -> dict:
    path = Path(path)
    cfg = _DEFAULT.copy()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v

    # env overrides
    cfg["llm"]["model"] = os.getenv("OPENAI_MODEL", cfg["llm"]["model"])
    cfg["tts"]["voice_id"] = os.getenv("ELEVEN_VOICE_ID", cfg["tts"]["voice_id"])
    cfg["tts"]["model_id"] = os.getenv("ELEVEN_MODEL_ID", cfg["tts"]["model_id"])
    return cfg

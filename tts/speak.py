# tts/speak.py
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv

load_dotenv()


def _save_text_fallback(text: str, save_to: Path) -> Path:
    """If TTS is unavailable, save the text as a .txt next to the intended audio."""
    p = Path(save_to)
    p.parent.mkdir(parents=True, exist_ok=True)
    txt_path = p.with_suffix(".txt")
    txt_path.write_text(text or "", encoding="utf-8")
    return txt_path


def speak_text(
    text: str,
    save_to: Optional[Path] = None,
    play: bool = True,
    voice_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synthesize speech with ElevenLabs and optionally play it.

    - If ELEVEN_API_KEY is missing or the request fails, we degrade gracefully:
      we save a .txt fallback (if save_to provided) and return a note.
    - If `play` is False (or pygame can't be imported), we only save the file.

    Returns a small dict with info about the outcome.
    """
    result: Dict[str, Any] = {"played": False, "path": None, "note": ""}

    # Handle empty input early
    if not (text or "").strip():
        result["note"] = "No text to speak"
        # Still write an empty file if a target is specified
        if save_to:
            _save_text_fallback(text, Path(save_to))
            result["path"] = str(Path(save_to).with_suffix(".txt"))
        return result

    api_key = os.getenv("ELEVEN_API_KEY")
    # allow config override
    voice_id = voice_id or os.getenv("ELEVEN_VOICE_ID", "OYTbf65OHHFELVut7v2H")
    model_id = model_id or os.getenv("ELEVEN_MODEL_ID", "eleven_flash_v2")

    # If no API key, don't crash—save text fallback and move on
    if not api_key:
        if save_to:
            txt_path = _save_text_fallback(text, Path(save_to))
            result.update(
                {"path": str(txt_path), "note": "ELEVEN_API_KEY missing; saved text fallback"}
            )
        else:
            result["note"] = "ELEVEN_API_KEY missing; no audio playback"
        return result

    # Build request
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {"model_id": model_id, "text": text}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code != 200 or not resp.content:
            note = f"TTS HTTP {resp.status_code}: {resp.text[:200]}"
            if save_to:
                txt_path = _save_text_fallback(text, Path(save_to))
                result.update({"path": str(txt_path), "note": note})
            else:
                result["note"] = note
            return result

        # Decide where to save the mp3
        if save_to:
            save_path = Path(save_to)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tmp.close()
            save_path = Path(tmp.name)

        save_path.write_bytes(resp.content)
        result["path"] = str(save_path)

        # Playback (optional)
        if play:
            try:
                import pygame  # import lazily so headless envs don't fail

                pygame.mixer.init()
                pygame.mixer.music.load(str(save_path))
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                pygame.mixer.quit()
                result["played"] = True
            except Exception as e:
                result["note"] = f"Playback skipped: {e}"

        # If we created a temp file and didn't ask to persist, keep it anyway so logs have the artifact
        return result

    except Exception as e:
        note = f"TTS exception: {e}"
        if save_to:
            txt_path = _save_text_fallback(text, Path(save_to))
            result.update({"path": str(txt_path), "note": note})
        else:
            result["note"] = note
        return result

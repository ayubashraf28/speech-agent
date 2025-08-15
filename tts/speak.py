# tts/speak.py
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv

load_dotenv()

# lightweight, Windows-friendly playback
try:
    from playsound import playsound as _playsound
except Exception:
    _playsound = None

def _play(path: Path) -> bool:
    # Try playsound, then pygame as a last resort.
    if _playsound is not None:
        try:
            _playsound(str(path), block=True)
            return True
        except Exception:
            pass
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        pygame.mixer.quit()
        return True
    except Exception:
        return False

def _save_text_fallback(text: str, save_to: Path) -> Path:
    p = Path(save_to)
    p.parent.mkdir(parents=True, exist_ok=True)
    txt_path = p.with_suffix(".txt")
    txt_path.write_text(text or "", encoding="utf-8")
    return txt_path

def _ensure_target_path(save_to: Optional[Path], suffix: str) -> Path:
    if save_to:
        p = Path(save_to).with_suffix(suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    return Path(tmp.name)

def _elevenlabs_tts(text: str, voice_id: str, model_id: str, api_key: str) -> bytes:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"}
    payload = {"model_id": model_id, "text": text}
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    if resp.status_code != 200 or not resp.content:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.content

def _gtts_mp3(text: str) -> bytes:
    # gTTS returns a file; write to temp and read back for a clean interface
    from gtts import gTTS
    t = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    t.close()
    gTTS(text).save(t.name)
    data = Path(t.name).read_bytes()
    try:
        Path(t.name).unlink(missing_ok=True)
    except Exception:
        pass
    return data

def _pyttsx3_wav(text: str, out_path: Path) -> None:
    import pyttsx3
    engine = pyttsx3.init()
    # (Optional) tweak voice/rate if you want:
    # engine.setProperty('rate', 185)
    engine.save_to_file(text, str(out_path))
    engine.runAndWait()

def speak_text(
    text: str,
    save_to: Optional[Path] = None,
    play: bool = True,
    voice_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Primary: ElevenLabs (MP3)
    Fallback 1: gTTS (MP3)
    Fallback 2: pyttsx3 offline (WAV)
    Returns: {'played': bool, 'path': str|None, 'note': str}
    """
    result: Dict[str, Any] = {"played": False, "path": None, "note": ""}

    if not (text or "").strip():
        result["note"] = "No text to speak"
        if save_to:
            _save_text_fallback(text, Path(save_to))
            result["path"] = str(Path(save_to).with_suffix(".txt"))
        return result

    # Configuration
    api_key = os.getenv("ELEVEN_API_KEY")
    voice_id = voice_id or os.getenv("ELEVEN_VOICE_ID", "OYTbf65OHHFELVut7v2H")
    model_id = model_id or os.getenv("ELEVEN_MODEL_ID", "eleven_flash_v2")

    # Try ElevenLabs first (if key present)
    if api_key:
        try:
            audio = _elevenlabs_tts(text, voice_id, model_id, api_key)
            mp3_path = _ensure_target_path(save_to, ".mp3")
            mp3_path.write_bytes(audio)
            result["path"] = str(mp3_path)
            if play:
                result["played"] = _play(mp3_path)
            return result
        except Exception as e:
            note = f"ElevenLabs failed: {e}"
            # If this is a quota/401/429 style error, we’ll fall through to gTTS/pyttsx3
            result["note"] = note

    # Fallback 1: gTTS (internet, no key)
    try:
        audio = _gtts_mp3(text)
        mp3_path = _ensure_target_path(save_to, ".mp3")
        mp3_path.write_bytes(audio)
        result["path"] = str(mp3_path)
        if play:
            result["played"] = _play(mp3_path)
        if result["note"]:
            result["note"] += " | used gTTS"
        else:
            result["note"] = "used gTTS"
        return result
    except Exception as e:
        # continue to offline fallback
        if result["note"]:
            result["note"] += f" | gTTS failed: {e}"
        else:
            result["note"] = f"gTTS failed: {e}"

    # Fallback 2: pyttsx3 (offline, WAV)
    try:
        wav_path = _ensure_target_path(save_to, ".wav")
        _pyttsx3_wav(text, wav_path)
        result["path"] = str(wav_path)
        if play:
            result["played"] = _play(wav_path)
        if result["note"]:
            result["note"] += " | used pyttsx3"
        else:
            result["note"] = "used pyttsx3"
        return result
    except Exception as e:
        # final fallback: save .txt
        if save_to:
            txt_path = _save_text_fallback(text, Path(save_to))
            result.update({"path": str(txt_path), "note": f"{result['note']} | offline TTS failed: {e}" if result["note"] else f"offline TTS failed: {e}"})
        else:
            result["note"] = f"{result['note']} | offline TTS failed: {e}" if result["note"] else f"offline TTS failed: {e}"
        return result

# asr/transcribe.py
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
from dotenv import load_dotenv

load_dotenv()

_model = None
_device = None


def _load_whisper():
    """Load Whisper once, using config-driven model selection."""
    global _model, _device
    if _model is not None:
        return _model, _device

    import torch, whisper

    # Configurable model via env
    model_name = os.getenv("ASR_WHISPER_MODEL", "small")
    preferred = "cuda" if torch.cuda.is_available() else "cpu"

    for dev in ([preferred, "cpu"] if preferred == "cuda" else ["cpu"]):
        try:
            print(f"[ASR] Loading Whisper '{model_name}' on {dev}")
            m = whisper.load_model(model_name, device=dev)
            _model, _device = m, dev
            return _model, _device
        except Exception as e:
            print(f"[ASR] Failed to load {model_name} on {dev}: {e}")

    raise RuntimeError(f"Whisper model '{model_name}' failed to load on both CUDA and CPU.")


def record_audio(duration: int = 7, samplerate: int = 16000):
    """Record live audio from default microphone."""
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    print("Recording complete.")
    return np.squeeze(audio)


def _save_wav(audio: np.ndarray, samplerate=16000) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, samplerate)
    return tmp.name


def transcribe_audio(audio: np.ndarray, samplerate: int = 16000) -> str:
    """Transcribe from an audio numpy array."""
    model, device = _load_whisper()
    wav_path = _save_wav(audio, samplerate)
    print("Transcribing...")
    result = model.transcribe(wav_path, fp16=(device == "cuda"))
    print(f"Saved file: {wav_path}")
    return result.get("text", "").strip()


def transcribe_file(file_path: str) -> str:
    """Transcribe from an existing audio file path."""
    model, device = _load_whisper()
    print(f"Transcribing existing file: {file_path}")
    result = model.transcribe(file_path, fp16=(device == "cuda"))
    return result.get("text", "").strip()

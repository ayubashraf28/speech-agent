# asr/transcribe.py
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile

# Lazy-loaded globals
_model = None
_device = None

def _load_whisper():
    """Load Whisper once; prefer CUDA but fall back to CPU cleanly."""
    global _model, _device
    if _model is not None:
        return _model, _device

    import torch, whisper
    preferred = "cuda" if torch.cuda.is_available() else "cpu"
    for dev in [preferred, "cpu"] if preferred == "cuda" else ["cpu"]:
        try:
            print(f"[ASR] Loading Whisper 'small' on {dev}")
            m = whisper.load_model("small", device=dev)
            _model, _device = m, dev
            return _model, _device
        except Exception as e:
            print(f"[ASR] Failed to load on {dev}: {e}")

    raise RuntimeError("Whisper model failed to load on both CUDA and CPU.")

def record_audio(duration=7, samplerate=16000):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    print("Recording complete.")
    return np.squeeze(audio)

def _save_wav(audio: np.ndarray, samplerate=16000) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, samplerate)
    return tmp.name

def transcribe_audio(audio, samplerate=16000):
    model, device = _load_whisper()
    wav_path = _save_wav(audio, samplerate)
    print("Transcribing...")
    result = model.transcribe(wav_path, fp16=(device == "cuda"))
    print(f"Saved file: {wav_path}")
    return result["text"]

def transcribe_file(file_path: str):
    model, device = _load_whisper()
    print(f"Transcribing existing file: {file_path}")
    result = model.transcribe(file_path, fp16=(device == "cuda"))
    return result["text"]

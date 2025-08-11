# asr/transcribe.py
import whisper
import sounddevice as sd
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path

# Load Whisper model once (small for speed, can change to "medium" or "large")
model = whisper.load_model("small")

def record_audio(duration=7, samplerate=16000):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Blocks until recording finishes
    print("Recording complete.")
    return np.squeeze(audio)

def save_wav(audio: np.ndarray, samplerate=16000, out_path: str = None) -> str:
    """Save numpy audio to a .wav file and return the path."""
    if out_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out_path = tmp.name
    sf.write(out_path, audio, samplerate)
    return out_path

def transcribe_audio(audio, samplerate=16000):
    """Transcribe audio (numpy array) with Whisper and return text."""
    wav_path = save_wav(audio, samplerate)
    print("Transcribing...")
    result = model.transcribe(wav_path)
    print(f"Saved file: {wav_path}")
    return result["text"]

def transcribe_file(file_path: str):
    """Transcribe existing audio file directly."""
    print(f"Transcribing existing file: {file_path}")
    result = model.transcribe(file_path)
    return result["text"]

if __name__ == "__main__":
    # Example usage: record and transcribe
    audio = record_audio()
    text = transcribe_audio(audio)
    print("You said:", text)

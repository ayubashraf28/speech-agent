
import whisper
import sounddevice as sd
import numpy as np
import tempfile
import soundfile as sf

model = whisper.load_model("small")  

def record_audio(duration=7, samplerate=16000):
    print(f"🎙️ Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    print(f"✅ Recording done. Audio shape: {audio.shape}")
    return np.squeeze(audio)


def transcribe_audio(audio, samplerate=16000):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, samplerate)
        print("🔎 Transcribing...")
        result = model.transcribe(f.name)
        print("Saved file:", f.name)
        return result["text"]

if __name__ == "__main__":
    audio = record_audio()
    text = transcribe_audio(audio)
    print("📝 You said:", text)

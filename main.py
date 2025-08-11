import datetime
import soundfile as sf
from pathlib import Path
from asr.diarize import diarize_wav, save_segments_csv

from asr.transcribe import record_audio, transcribe_audio
from llm.generate import generate_response
from tts.speak import speak_text

def save_text(path, content):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def main():
    # session folder
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = Path(f"data/logs/{timestamp}")
    session_dir.mkdir(parents=True, exist_ok=True)

    # record
    print("[Main] recording…")
    audio = record_audio(duration=7, samplerate=16000)
    user_wav = session_dir / "user.wav"
    sf.write(user_wav, audio, 16000)
    print(f"[Main] saved audio -> {user_wav}")

    # diarization
    try:
        print("[Main] diarizing…")
        segments = diarize_wav(str(user_wav))
        save_segments_csv(segments, str(session_dir / "diarization.csv"))
        unique_speakers = {s.speaker for s in segments}
        print(f"[Main] diarization done: {len(segments)} turns, ~{len(unique_speakers)} speakers")
    except Exception as e:
        print(f"[Main] diarization skipped (error): {e}")


    # transcribe
    print("[Main] transcribing…")
    transcript = transcribe_audio(audio, samplerate=16000)
    print("You said:", transcript)
    save_text(session_dir / "transcript.txt", transcript)

    # LLM reply
    print("[Main] generating reply…")
    reply = generate_response(transcript)
    print("GPT says:", reply)
    save_text(session_dir / "response.txt", reply)

    # TTS (save mp3 in session folder to avoid temp-file lock)
    print("[Main] speaking…")
    tts_mp3 = session_dir / "response.mp3"
    speak_text(reply, save_to=tts_mp3)
    print(f"[Main] saved TTS -> {tts_mp3}")

    # append session summary
    with open("data/logs/session_summary.log", "a", encoding="utf-8") as f:
        f.write(f"{timestamp} | USER: {transcript} | AGENT: {reply}\n")

    print("[Main] done ✅")

if __name__ == "__main__":
    main()

import os
import datetime
import soundfile as sf

from asr.transcribe import record_audio, transcribe_audio
from llm.generate import generate_response
from tts.speak import speak_text

def save_text(filename, content):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    # Create a logs folder if it doesn't exist
    os.makedirs("data/logs", exist_ok=True)

    #Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    #Record audio
    audio = record_audio(duration=7)
    audio_filename = f"data/logs/{timestamp}_user.wav"
    sf.write(audio_filename, audio, 16000)

    #Transcribe
    transcript = transcribe_audio(audio)
    print("You said:", transcript)
    transcript_filename = f"data/logs/{timestamp}_transcript.txt"
    save_text(transcript_filename, transcript)

    #Generate GPT response
    reply = generate_response(transcript)
    print("GPT says:", reply)
    reply_filename = f"data/logs/{timestamp}_response.txt"
    save_text(reply_filename, reply)

    #Speak the response
    speak_text(reply)

    #Append summary log (optional)
    summary_log = "data/logs/session_summary.log"
    with open(summary_log, 'a', encoding='utf-8') as f:
        f.write(f"{timestamp} | USER: {transcript} | AGENT: {reply}\n")

if __name__ == "__main__":
    main()

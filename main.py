from asr.transcribe import record_audio, transcribe_audio
from llm.generate import generate_response
from tts.speak import speak_text

def main():
    # 1. Record + transcribe
    audio = record_audio(duration=7)
    text = transcribe_audio(audio)
    print("You said:", text)

    # 2. LLM generates reply
    reply = generate_response(text)
    print("GPT says:", reply)

    # 3. Convert to TTS
    speak_text(reply)

if __name__ == "__main__":
    main()

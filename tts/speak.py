# tts/speak.py
import os
import requests
import tempfile
import pygame
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "OYTbf65OHHFELVut7v2H")  # default voice
ELEVEN_MODEL_ID = os.getenv("ELEVEN_MODEL_ID", "eleven_flash_v2")       # default model

if not ELEVEN_API_KEY:
    raise RuntimeError("ELEVEN_API_KEY not set in .env")

def speak_text(text: str, save_to: Path = None):
    """
    Convert text to speech using ElevenLabs API and play it.
    
    Args:
        text (str): The text to be spoken.
        save_to (Path): Optional path to save the audio file.
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"

    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }

    payload = {
        "model_id": ELEVEN_MODEL_ID,
        "text": text
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200 or not response.content:
            print(f"[TTS] Error {response.status_code}: {response.text}")
            return

        # Save either to temp file or provided path
        if save_to:
            save_path = Path(save_to)
            save_path.write_bytes(response.content)
        else:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tmp_file.write(response.content)
            tmp_file.flush()
            save_path = Path(tmp_file.name)

        # Play the audio
        pygame.mixer.init()
        pygame.mixer.music.load(str(save_path))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)

        pygame.mixer.quit()

        # Remove temp file if we didn't want to keep it
        if not save_to:
            try:
                os.unlink(save_path)
            except PermissionError:
                print(f"[TTS] Could not delete temp file: {save_path}")

    except Exception as e:
        print(f"[TTS] Exception occurred: {e}")

if __name__ == "__main__":
    speak_text("Hi! I am your speech agent.")

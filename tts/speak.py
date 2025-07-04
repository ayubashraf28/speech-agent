import os
import requests
import tempfile
import pygame
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ELEVEN_API_KEY")
print("API key loaded:", api_key)

def speak_text(text: str):
    url = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    voice_id = "OYTbf65OHHFELVut7v2H"   # Rachel or your chosen ID
    model_id = "eleven_flash_v2"

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }

    payload = {
        "model_id": model_id,
        "text": text
    }

    response = requests.post(url.format(voice_id=voice_id), json=payload, headers=headers)
    print("Status code:", response.status_code)
    print("Response content length:", len(response.content))

    if response.status_code == 200 and len(response.content) > 0:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        try:
            temp.write(response.content)
            temp.flush()
            temp.close()   
            pygame.mixer.init()
            pygame.mixer.music.load(temp.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                continue
        finally:
            
            os.unlink(temp.name)
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    speak_text("Hi! I am your speech agent.")

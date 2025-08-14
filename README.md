Speech Agent – Real-Time Speech → LLM → TTS with Optional Speaker Diarization
Overview
This project implements a modular speech agent capable of:

Recording or loading audio

Optional speaker diarization (multi-speaker segmentation)

Transcription via Whisper

Context-aware text generation via OpenAI GPT

Text-to-speech playback via ElevenLabs

Full session logging for reproducibility

It’s designed with a production-ready structure so you can easily extend, swap models, or integrate into larger systems.

Features
Record or load audio files (.wav/.mp3)

Speaker-aware prompting (when diarization enabled)

Whisper ASR with configurable model/device

LLM response generation via OpenAI API

TTS playback + MP3 saving via ElevenLabs

Full logging: transcript, response, diarization CSV, audio files, session summary

Central config.yaml for all parameters

Installation
Clone repo & enter directory

git clone https://github.com/yourusername/speech-agent.git
cd speech-agent
Create & activate virtual environment

python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
Install dependencies

pip install -r requirements.txt
Install FFmpeg (required for Whisper)

Windows: Download from https://www.gyan.dev/ffmpeg/builds/ and add bin/ to PATH

Mac: brew install ffmpeg

Linux: sudo apt install ffmpeg

Set up .env file

env
OPENAI_API_KEY=your_openai_api_key
ELEVEN_API_KEY=your_elevenlabs_api_key
HUGGINGFACE_TOKEN=your_hf_token
Configuration

Edit config.yaml to control:

app:
  diarization: true        # Enable/disable diarization
  samplerate: 16000
  duration: 7
llm:
  model: gpt-3.5-turbo
  base_system_prompt: "You are a helpful speech agent."
tts:
  voice_id: "OYTbf65OHHFELVut7v2H"
  model_id: "eleven_flash_v2"
asr:
  whisper_model: "small"
runtime:
  device: "auto"           # auto / cpu / cuda

Usage
1. Record & process new audio
python main.py

2. Process existing audio file
python main.py --file data/test/demo.wav

3. Disable diarization
python main.py --no-diarize

4. Adjust recording duration
python main.py --duration 5

Output
Each session saves to:

data/logs/<timestamp>/
    user.wav          # Original audio
    transcript.txt    # ASR output
    response.txt      # GPT reply
    response.mp3      # TTS output
    diarization.csv   # Speaker segments (if enabled)
    run.log           # Detailed log
A rolling data/logs/session_summary.log file keeps a one-line record per session.
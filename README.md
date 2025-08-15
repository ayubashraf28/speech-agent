MULTI-PARTY CONVERSATIONAL SPEECH AGENT

A real-time speech agent that listens on your microphone, detects utterance boundaries (VAD), transcribes speech (ASR), generates a reply (LLM), and optionally speaks the reply (TTS). It can label speakers across turns (Resemblyzer embeddings), keep short-term per-speaker memory, and persist a speaker database between sessions.

WHY THIS EXISTS

Group conversations are messy: interruptions, quick turn-taking, and topic switches.

This project focuses on the plumbing required for live, multi-speaker interaction:

Non-blocking audio capture so the loop stays responsive.

Simple, robust VAD.

Lightweight speaker identification with names learned on the fly (“my name is Alice”).

Per-speaker conversational memory so responses stay contextual.

FEATURES

Real-time mic capture via sounddevice.InputStream (callback → Queue).

RMS VAD with configurable frame length and end-of-silence.

ASR in asr/transcribe.py (e.g., Whisper).

LLM reply in llm/generate.py (speaker-aware system prompt).

TTS in tts/speak.py (ElevenLabs if API key present; otherwise saves a .txt fallback).

Optional Speaker ID in utils/speaker_id.py (Resemblyzer or simple MFCC fallback).

Per-speaker short-term memory in utils/memory.py.

Speaker DB persistence to data/speakers/default.npz; list/reset from CLI.

PROJECT STRUCTURE

.
├─ realtime_agent.py Main entrypoint (VAD, loop, ASR → LLM → TTS, SpeakerID, Memory)
├─ config.py load_config(): loads settings (e.g., base prompt/model)
├─ asr/
│ └─ transcribe.py transcribe_file(path) → text
├─ llm/
│ └─ generate.py generate_response(text, system_prompt, model_name) → reply
├─ tts/
│ └─ speak.py speak_text(text, save_to, play, voice_id, model_id) → dict
├─ utils/
│ ├─ speaker_id.py SpeakerID: enroll/identify, thresholds, persistence
│ └─ memory.py SpeakerMemory: rolling per-speaker history
├─ data/
│ ├─ live/ captured utterances + agent replies
│ └─ speakers/
│ └─ default.npz speaker embedding database (created at runtime)
├─ requirements.txt
└─ README.md (this file)

HOW THE PIPELINE WORKS

Microphone (callback) → frames enter a Queue → VAD groups frames into an utterance → save .wav →
ASR produces text → (optional) SpeakerID identifies/enrolls speaker →
build speaker-aware LLM prompt (with per-speaker short memory) → LLM reply →
TTS generates audio (or writes a text fallback) → loop continues.

REQUIREMENTS

Python 3.9+ (tested on Windows 10/11)

A working microphone (list devices with: python -m sounddevice)

ASR: whatever asr/transcribe.py is configured to use (e.g., Whisper + PyTorch)

TTS: ElevenLabs (set ELEVEN_API_KEY); otherwise the code writes a .txt fallback

SpeakerID (optional but recommended):

resemblyzer==0.1.1.dev0 (and its deps: librosa, webrtcvad, etc.)

On Windows you may need Microsoft C++ Build Tools for webrtcvad

CUDA optional (Resemblyzer will auto-select CPU/GPU)

If Resemblyzer is unavailable, the code tries a simpler MFCC embedding via torchaudio; if neither is present, SpeakerID is disabled automatically.

SETUP

Create and activate a virtual environment
Windows:
python -m venv venv
venv\Scripts\activate
macOS/Linux:
python -m venv venv
source venv/bin/activate

Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
(Optional) Speaker-ID extras (if not already installed):
pip install resemblyzer==0.1.1.dev0 librosa==0.10.1 webrtcvad==2.0.10

Environment variables for TTS (create a .env file if you don’t have one)
ELEVEN_API_KEY=YOUR_KEY
ELEVEN_VOICE_ID=OPTIONAL (defaults in code)
ELEVEN_MODEL_ID=eleven_flash_v2 (default)

(Optional) Configure LLM in config.yaml, e.g.:
llm:
model: gpt-3.5-turbo
base_system_prompt: "You are a helpful, concise multi-party assistant..."

QUICK START

Find your input device index:
python -m sounddevice
(note the index of your microphone)

Run the agent (Windows example, change index as needed):
python realtime_agent.py ^
--device 12 ^
--samplerate 48000 ^
--frame-ms 20 ^
--end-silence-ms 1500 ^
--spk-id ^
--speaker-db data/speakers/default.npz ^
--spk-threshold 0.86 ^
--spk-margin 0.06 ^
--mem-turns 3

Speak naturally. To help labeling, say: “My name is Alice” (once).
The agent prints transcripts and replies and saves artifacts to data/live/:

utter_YYYY-MM-DD_HH-MM-SS.wav (your speech)

utter_...reply.mp3 (agent TTS) or utter..._reply.txt if TTS fallback was used

List or reset speakers:
List DB:
python realtime_agent.py --spk-id --speaker-db data/speakers/default.npz --list-speakers
Reset DB:
python realtime_agent.py --spk-id --speaker-db data/speakers/default.npz --reset-speakers

SPEAKER LABELING (HOW IT WORKS)

Name extraction: if a transcript contains “my name is X / I’m X / call me X”, we enroll that utterance under X.

Embeddings: later utterances are embedded and cosine-matched to known centroids.

Thresholds:

If similarity ≥ --spk-threshold, accept best match (and update centroid by EMA).

If similarity is borderline within --spk-margin of the threshold and equals the last label, we keep the last label to avoid flip-flop.

Multiple low-confidence hits are required before minting a new Guest-N (hysteresis is inside SpeakerID).

Persistence: DB at data/speakers/default.npz is saved/loaded automatically.

Tuning tips:

Start with --spk-threshold 0.86 and --spk-margin 0.06.

If you get too many new Guest-N, lower threshold (e.g., 0.84).

If it confuses two speakers, raise threshold (e.g., 0.88–0.90).

VAD TUNING

--frame-ms: analysis frame size; 20–30 ms is typical.

--end-silence-ms: required silence to end an utterance; 1200–1800 ms works well.

Increase end-silence if it chops sentences early; decrease if it lags.

PER-SPEAKER MEMORY

utils/memory.SpeakerMemory(max_turns=N) stores the last N user/agent turns per label.

The LLM receives “Recent context for {label}” when available.

Configure with --mem-turns (e.g., 3).

TTS

tts/speak.py uses ElevenLabs when ELEVEN_API_KEY is set.

If the API key is missing or quota is exceeded, the function does not crash: it writes a ..._reply.txt file and logs a note.

Pass --no-play to suppress playing audio through speakers (still saves mp3 or txt).

Quick check for TTS:
python -c "from tts.speak import speak_text; print(speak_text('Hello!', play=True))"

CORPUS CAPTURE (FOR SUBMISSION)

Record multiple short sessions (5–10 minutes) with two speakers.

Each speaker should say their name once at the start.

Alternate turns and include some quick back-and-forth.

Artifacts are in data/live/.

(Optional) Create a simple index CSV if you add a small script:
python scripts/make_corpus_index.py
→ data/corpus_index.csv

Zip the repository minus very large caches if required by submission rules.

TROUBLESHOOTING

Stuck at “Calibrating noise floor” or silence:
Probably wrong device index or mic permissions. Run python -m sounddevice and pick the correct input index.

Noise RMS≈0.0000 (very low threshold):
Likely muted/incorrect device. Re-select your actual mic.

TTS not speaking:
Check ELEVEN_API_KEY. HTTP 401 about quota means your plan is out of credits; the code will save a .txt reply instead.

Speaker labels keep changing:
Raise --spk-threshold or lower --spk-margin. Ensure each speaker says their name once.

Resemblyzer install issues on Windows:
Install Microsoft C++ Build Tools, then reinstall packages. Or run without --spk-id.

Whisper/CUDA issues:
Ensure asr/transcribe.py can fall back to CPU or choose an appropriate model there.

REPRO COMMANDS (KNOWN GOOD)
enumerate devices

python -m sounddevice

run the agent

python realtime_agent.py ^
--device 12 ^
--samplerate 48000 ^
--frame-ms 20 ^
--end-silence-ms 1500 ^
--spk-id ^
--speaker-db data/speakers/default.npz ^
--spk-threshold 0.86 ^
--spk-margin 0.06 ^
--mem-turns 3

CONFIGURATION NOTES

config.yaml (optional):
llm:
model: gpt-3.5-turbo
base_system_prompt: >
You are a concise, helpful multi-party assistant. Keep replies to 1–3 sentences.
Keep each thread separate; do not mix details across speakers.

tts (in code or via .env):
ELEVEN_API_KEY=...
ELEVEN_VOICE_ID=... (optional)
ELEVEN_MODEL_ID=eleven_flash_v2 (default)

ACKNOWLEDGEMENTS

ASR typically powered by Whisper (or your configured backend in asr/transcribe.py).

Embeddings by Resemblyzer when available.

Audio I/O via python-sounddevice.

LICENSE

This project includes third-party components under their respective licenses. See requirements.txt and upstream repositories for details.

TL;DR TO RUN

python -m sounddevice → pick mic index

set .env with ElevenLabs key (optional for voice)

run:
python realtime_agent.py --device <IDX> --samplerate 48000 --frame-ms 20 --end-silence-ms 1500 --spk-id --speaker-db data/speakers/default.npz --spk-threshold 0.86 --spk-margin 0.06 --mem-turns 3

Say: “My name is Alice…”, “My name is Ryan…”, then alternate questions.
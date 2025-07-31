from pyannote.audio import Pipeline
import os
from dotenv import load_dotenv

# Disable symlink-based caching
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SPEECHBRAIN_STRATEGY"] = "copy"

load_dotenv()

# Load Hugging Face token from .env
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# Load pipeline with your token
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=hf_token
)

# Path to audio file
audio_file = "data/test/demo.wav"

# Run diarization
diarization = pipeline(audio_file)

# Print segments and speaker labels
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}s - {turn.end:.1f}s: Speaker {speaker}")


import os
from dataclasses import dataclass
from typing import List
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SPEECHBRAIN_STRATEGY"] = "copy"

load_dotenv()

@dataclass
class Segment:
    start: float
    end: float
    speaker: str

_pipeline = None

def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        from pyannote.audio import Pipeline   # <-- lazy import
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise RuntimeError("HUGGINGFACE_TOKEN not found in .env")
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=hf_token
        )
    return _pipeline

def diarize_wav(audio_path: str) -> List[Segment]:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    pipeline = _load_pipeline()
    diarization = pipeline(str(audio_path))
    return [
        Segment(start=float(turn.start), end=float(turn.end), speaker=str(speaker))
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

def save_segments_csv(segments: List[Segment], csv_path: str):
    import pandas as pd
    pd.DataFrame([s.__dict__ for s in segments]).to_csv(csv_path, index=False)
    print(f"[Diarize] CSV saved: {csv_path}")

if __name__ == "__main__":
    # Test run
    test_file = "data/test/demo.wav"
    results = diarize_wav(test_file)
    for seg in results:
        print(f"{seg.start:.1f}s - {seg.end:.1f}s: {seg.speaker}")
    save_segments_csv(results, "data/test/demo_diarization.csv")

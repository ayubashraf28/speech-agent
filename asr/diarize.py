# diarize.py
import os
from dataclasses import dataclass, asdict
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# Avoid various hub warnings
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
    """
    Lazy-load the pyannote diarization pipeline.
    Uses HUGGINGFACE_TOKEN or HF_TOKEN. Optional DIARIZE_DEVICE (cpu|cuda|cuda:0).
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "No Hugging Face token found. Set HUGGINGFACE_TOKEN (or HF_TOKEN) in your environment."
        )

    # Optional explicit device selection
    device = os.getenv("DIARIZE_DEVICE")  # e.g., "cpu", "cuda", "cuda:0"

    # Lazy import so environments without pyannote don't crash at import time
    from pyannote.audio import Pipeline

    kwargs = {"use_auth_token": hf_token}
    if device:
        kwargs["device"] = device

    _pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", **kwargs)
    return _pipeline


def diarize_wav(audio_path: str) -> List[Segment]:
    """
    Run diarization on an audio file and return a flat list of segments.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    pipeline = _load_pipeline()
    diarization = pipeline(str(audio_path))
    segments: List[Segment] = [
        Segment(start=float(turn.start), end=float(turn.end), speaker=str(speaker))
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    return segments


def merge_contiguous(segments: List[Segment], gap_threshold: float = 0.2, min_duration: float = 0.0) -> List[Segment]:
    """
    Merge back-to-back segments from the same speaker when the gap is small.

    gap_threshold: maximum gap (seconds) to consider merging contiguous segments.
    min_duration: drop merged segments shorter than this duration.
    """
    if not segments:
        return []

    # Ensure sorted by start time
    segs = sorted(segments, key=lambda s: (s.start, s.end))

    merged: List[Segment] = []
    cur = segs[0]

    for nxt in segs[1:]:
        if nxt.speaker == cur.speaker and (nxt.start - cur.end) <= gap_threshold:
            # extend current segment
            cur = Segment(start=cur.start, end=max(cur.end, nxt.end), speaker=cur.speaker)
        else:
            if (cur.end - cur.start) >= min_duration:
                merged.append(cur)
            cur = nxt
    # append last
    if (cur.end - cur.start) >= min_duration:
        merged.append(cur)

    return merged


def save_segments_csv(segments: List[Segment], csv_path: str):
    pd.DataFrame([asdict(s) for s in segments]).to_csv(csv_path, index=False)
    print(f"[Diarize] CSV saved: {csv_path}")


def save_rttm(segments: List[Segment], rttm_path: str, uri: Optional[str] = "audio"):
    """
    Save segments in RTTM format (useful for tooling).
    """
    lines = []
    for s in segments:
        dur = max(0.0, s.end - s.start)
        lines.append(f"SPEAKER {uri} 1 {s.start:.3f} {dur:.3f} <NA> <NA> {s.speaker} <NA> <NA>")
    Path(rttm_path).parent.mkdir(parents=True, exist_ok=True)
    Path(rttm_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[Diarize] RTTM saved: {rttm_path}")


if __name__ == "__main__":
    # Quick sanity test (adjust path)
    test_file = "data/test/demo.wav"
    try:
        results = diarize_wav(test_file)
        print(f"Raw segments: {len(results)}")
        merged = merge_contiguous(results, gap_threshold=0.2, min_duration=0.25)
        print(f"Merged segments: {len(merged)}")
        save_segments_csv(merged, "data/test/demo_diarization.csv")
        save_rttm(merged, "data/test/demo_diarization.rttm", uri="demo")
        for seg in merged[:10]:
            print(f"{seg.start:.2f}–{seg.end:.2f} {seg.speaker}")
    except Exception as e:
        print(f"[Diarize] Error in test run: {e}")

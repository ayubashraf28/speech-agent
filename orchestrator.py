# orchestrator.py
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable, Optional, Tuple
import math
import soundfile as sf

__all__ = [
    "Turn",
    "group_contiguous_by_speaker",
    "slice_wav",
    "make_turns_from_diarization",
    "format_tagged_transcript",
]


@dataclass
class Turn:
    speaker: str
    start: float  # seconds
    end: float    # seconds
    wav_path: Path  # segment audio path

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))


def _normalize_segments(
    segments: Iterable[Tuple[float, float, str]]
) -> List[Tuple[float, float, str]]:
    """
    Ensure (start, end, speaker) are well-formed and sorted by start.
    Drops segments with non-positive duration.
    """
    clean: List[Tuple[float, float, str]] = []
    for s, e, spk in segments:
        try:
            s = float(s)
            e = float(e)
        except Exception:
            continue
        if e <= s:
            continue
        clean.append((s, e, str(spk)))
    clean.sort(key=lambda x: (x[0], x[1]))
    return clean


def group_contiguous_by_speaker(
    segments: Iterable[Tuple[float, float, str]],
    gap: float = 0.3,
    min_duration: float = 0.0,
) -> List[Tuple[float, float, str]]:
    """
    Merge consecutive diarization segments by same speaker when the gap is small.

    Args:
        segments: iterable of (start, end, speaker)
        gap: merge if next.start - prev.end <= gap (seconds)
        min_duration: drop merged segments shorter than this duration (seconds)

    Returns:
        List of merged (start, end, speaker) tuples.
    """
    segs = _normalize_segments(segments)
    if not segs:
        return []

    merged: List[List[float | str]] = []  # [start, end, speaker]
    for s, e, spk in segs:
        if not merged:
            merged.append([s, e, spk])
            continue
        ps, pe, pspk = merged[-1]
        if spk == pspk and (s - float(pe)) <= gap:
            merged[-1][1] = e  # extend
        else:
            # flush previous if long enough
            if (float(pe) - float(ps)) >= min_duration:
                pass  # keep as-is
            merged.append([s, e, spk])

    # Filter by min_duration
    out: List[Tuple[float, float, str]] = []
    for s, e, spk in merged:
        if (float(e) - float(s)) >= min_duration:
            out.append((float(s), float(e), str(spk)))
    return out


def slice_wav(in_wav: Path, out_wav: Path, start_s: float, end_s: float) -> int:
    """
    Write a slice of in_wav [start_s, end_s) to out_wav, preserving sample rate.

    Uses soundfile's start/stop parameters to avoid loading the entire file.
    Returns:
        The sample rate used for the output file.
    """
    in_wav = Path(in_wav)
    out_wav = Path(out_wav)
    assert in_wav.exists(), f"Input WAV not found: {in_wav}"

    with sf.SoundFile(str(in_wav)) as f:
        sr = int(f.samplerate)
        start_i = max(0, int(math.floor(start_s * sr)))
        end_i = min(len(f), int(math.ceil(end_s * sr)))
        frames = max(0, end_i - start_i)

    if frames <= 0:
        # Create a tiny silent file to keep turn indexing stable
        with sf.SoundFile(str(in_wav)) as f:
            sr = int(f.samplerate)
        sf.write(str(out_wav), [], sr)
        return sr

    data, sr = sf.read(str(in_wav), start=start_i, stop=end_i, dtype="float32", always_2d=False)
    # stereo -> mono
    if hasattr(data, "ndim") and getattr(data, "ndim", 1) == 2:
        data = data[:, 0]
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), data, sr)
    return sr


def make_turns_from_diarization(
    in_wav: Path,
    diarization_rows: List[Tuple[float, float, str]],
    turns_dir: Path,
    gap: float = 0.3,
    min_turn_duration: float = 0.0,
) -> List[Turn]:
    """
    From diarization rows (start,end,speaker), create merged turns and write per-turn WAV files.

    Args:
        in_wav: path to full session audio
        diarization_rows: list of (start, end, speaker)
        turns_dir: output directory for turn wavs
        gap: merge gap threshold in seconds (see group_contiguous_by_speaker)
        min_turn_duration: drop turns shorter than this

    Returns:
        List[Turn] with paths to the sliced audio for each merged speaker turn.
    """
    turns_dir = Path(turns_dir)
    turns_dir.mkdir(parents=True, exist_ok=True)

    merged = group_contiguous_by_speaker(diarization_rows, gap=gap, min_duration=min_turn_duration)
    turns: List[Turn] = []

    for i, (start, end, spk) in enumerate(merged, 1):
        out_wav = turns_dir / f"turn_{i:02d}_{spk}.wav"
        slice_wav(in_wav, out_wav, start, end)
        t = Turn(speaker=spk, start=float(start), end=float(end), wav_path=out_wav)
        # guard against accidental zero-length
        if t.duration <= 0.0:
            continue
        turns.append(t)

    return turns


def format_tagged_transcript(
    turns: List[Turn],
    turn_texts: List[str],
    show_timestamps: bool = True,
) -> str:
    """
    Build a speaker-tagged transcript once you have per-turn ASR text.

    Args:
        turns: List of Turn objects (same order you transcribed)
        turn_texts: ASR outputs aligned to turns (len must match)
        show_timestamps: include [start–end] timestamps per line

    Returns:
        A multi-line string, e.g.:
          [spk_00 00:01.2–03:18.7] Hello...
          [spk_01 03:19.0–05:07.1] Hi there...
    """
    assert len(turns) == len(turn_texts), "turns and turn_texts must have the same length"

    def fmt_time(x: float) -> str:
        return f"{x:06.1f}".rjust(6)

    lines: List[str] = []
    for t, text in zip(turns, turn_texts):
        label = t.speaker
        if show_timestamps:
            lines.append(f"[{label} {fmt_time(t.start)}–{fmt_time(t.end)}] {text.strip()}")
        else:
            lines.append(f"[{label}] {text.strip()}")
    return "\n".join(lines)

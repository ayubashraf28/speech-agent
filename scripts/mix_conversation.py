# mix_conversation.py
import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

from pydub import AudioSegment


def _check_ffmpeg() -> None:
    """Warn if ffmpeg/avlib might be missing for pydub backends."""
    if shutil.which("ffmpeg") is None and shutil.which("avconv") is None:
        print("[mix] Warning: ffmpeg/avconv not found in PATH. Some formats may fail to load.")


def _load_index(folder: Path, index_name: str) -> List[Dict[str, Any]]:
    idx_path = folder / index_name
    if not idx_path.exists():
        raise FileNotFoundError(f"Index file not found: {idx_path}")

    rows: List[Dict[str, Any]] = []
    with idx_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row["turn"] = int(row["turn"])
                row["start_s"] = float(row["start_s"])
                row["end_s"] = float(row["end_s"])
            except Exception as e:
                print(f"[mix] Skipping malformed row {row}: {e}")
                continue
            rows.append(row)
    rows.sort(key=lambda r: r["turn"])
    return rows


def _peak_normalize(seg: AudioSegment, target_dbfs: Optional[float]) -> AudioSegment:
    """Simple peak normalization towards target dBFS (None = no-op)."""
    if target_dbfs is None:
        return seg
    # Avoid division-by-zero on silent segments
    if seg.max_dBFS == float("-inf"):
        return seg
    change = target_dbfs - seg.max_dBFS
    return seg.apply_gain(change)


def build_insert_mix(
    session_dir: Path,
    original_audio: Path,
    index_name: str = "conversation_index.csv",
    gap_before_ms: int = 250,
    gap_after_ms: int = 200,
    crossfade_ms: int = 0,
    normalize_dbfs: Optional[float] = None,
    outdir: Optional[Path] = None,
) -> Path:
    """
    Create Q/A style mix: [Human turn] -> (gap_before) -> [Agent reply] -> (gap_after) -> next...

    Parameters
    ----------
    session_dir : Path
        Folder containing per-turn TTS files and the index CSV.
    original_audio : Path
        The original mixed human audio (wav/mp3, etc).
    index_name : str
        CSV file name with columns: turn,start_s,end_s,speaker,tts_file
    gap_before_ms : int
        Pause before agent reply (ms)
    gap_after_ms : int
        Pause after agent reply (ms)
    crossfade_ms : int
        Crossfade between concatenated segments (ms); 0 disables crossfade
    normalize_dbfs : Optional[float]
        If set, peak-normalize both human segments and TTS to this dBFS (e.g., -3.0)
    outdir : Optional[Path]
        Where to save outputs; defaults to session_dir

    Returns
    -------
    Path to the created mixed mp3.
    """
    _check_ffmpeg()
    index = _load_index(session_dir, index_name=index_name)

    if not original_audio.exists():
        raise FileNotFoundError(f"Original audio not found: {original_audio}")

    print(f"[mix] Loading original audio: {original_audio}")
    orig = AudioSegment.from_file(original_audio)

    mixed = AudioSegment.silent(duration=0)
    timeline: List[Dict[str, Any]] = []

    def add_seg(base: AudioSegment, seg: AudioSegment) -> AudioSegment:
        if crossfade_ms and len(base) > 0:
            return base.append(seg, crossfade=crossfade_ms)
        return base + seg

    for row in index:
        start_ms = int(row["start_s"] * 1000)
        end_ms = int(row["end_s"] * 1000)
        if end_ms <= start_ms:
            print(f"[mix] Skipping non-positive span turn={row.get('turn')}: {start_ms}..{end_ms}")
            continue

        # 1) Append the human slice
        human = orig[start_ms:end_ms]
        human = _peak_normalize(human, normalize_dbfs)
        mixed = add_seg(mixed, human)

        # 2) Gap before agent
        if gap_before_ms > 0:
            mixed = add_seg(mixed, AudioSegment.silent(duration=gap_before_ms))

        # 3) Append the agent reply (if present)
        tts_file = row.get("tts_file")
        if not tts_file:
            print(f"[mix] Missing tts_file in index row for turn {row.get('turn')}, skipping reply.")
        else:
            tts_path = session_dir / tts_file
            if tts_path.exists():
                try:
                    agent = AudioSegment.from_file(tts_path)
                    agent = _peak_normalize(agent, normalize_dbfs)
                    mixed = add_seg(mixed, agent)

                    # 4) Optional gap after
                    if gap_after_ms > 0:
                        mixed = add_seg(mixed, AudioSegment.silent(duration=gap_after_ms))

                    timeline.append(
                        {
                            "turn": row["turn"],
                            "speaker": row.get("speaker", ""),
                            "segment_start_ms": start_ms,
                            "segment_end_ms": end_ms,
                            "reply_file": tts_file,
                            "reply_duration_ms": int(len(agent)),
                        }
                    )
                except Exception as e:
                    print(f"[mix] Could not load TTS for turn {row['turn']} at {tts_path}: {e}")
            else:
                print(f"[mix] TTS file not found for turn {row['turn']}: {tts_path}")

    # Save outputs
    outdir = outdir or session_dir
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"mixed_conversation_{gap_before_ms}b_{gap_after_ms}a"
    if crossfade_ms:
        stem += f"_xf{crossfade_ms}"
    if normalize_dbfs is not None:
        stem += f"_norm{int(abs(normalize_dbfs))}"

    out_mp3 = outdir / f"{stem}.mp3"
    mixed.export(out_mp3, format="mp3")

    timeline_path = outdir / f"{stem}.json"
    with timeline_path.open("w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2)

    print(f"[mix] Wrote {out_mp3}")
    print(f"[mix] Wrote {timeline_path}")
    return out_mp3


def main():
    p = argparse.ArgumentParser(
        description="Create Q/A-style mix: [Human] → [gap] → [Agent reply] → [gap] → next"
    )
    p.add_argument("session_folder", type=Path, help="Path to session dir with index CSV and TTS files")
    p.add_argument("--original", type=Path, help="Path to original WAV/MP3 file (defaults to first audio in dir)")
    p.add_argument("--index", default="conversation_index.csv", help="Index CSV filename")
    p.add_argument("--gap-before-ms", type=int, default=250, help="Pause before agent reply (ms)")
    p.add_argument("--gap-after-ms", type=int, default=200, help="Pause after agent reply (ms)")
    p.add_argument("--crossfade-ms", type=int, default=0, help="Optional crossfade between segments (ms)")
    p.add_argument("--normalize-dbfs", type=float, default=None, help="Peak-normalize segments to target dBFS (e.g., -3.0)")
    p.add_argument("--outdir", type=Path, default=None, help="Output directory (defaults to session folder)")
    args = p.parse_args()

    session_dir = args.session_folder
    if not session_dir.exists():
        raise FileNotFoundError(session_dir)

    if args.original:
        original = args.original
    else:
        # best guess: first audio in session dir
        candidates = list(session_dir.glob("*.wav")) + list(session_dir.glob("*.mp3"))
        if not candidates:
            raise FileNotFoundError("Could not find original audio. Pass --original path.")
        original = candidates[0]

    build_insert_mix(
        session_dir=session_dir,
        original_audio=original,
        index_name=args.index,
        gap_before_ms=args.gap_before_ms,
        gap_after_ms=args.gap_after_ms,
        crossfade_ms=args.crossfade_ms,
        normalize_dbfs=args.normalize_dbfs,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()

# scripts/build_corpus.py
import argparse
import csv
import shutil
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

# Make repo imports work when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asr.transcribe import transcribe_file  # noqa
from utils.speaker_id import SpeakerID       # noqa
import re

def rms(x):
    import numpy as np
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x.astype(np.float32)))))

def extract_name(text: str):
    t = (text or "").strip()
    m = re.search(r"\b(i am|i'm|this is|my name is|call me)\s+([A-Za-z][A-Za-z\-']{1,30})\b", t, flags=re.I)
    if m:
        return m.group(2).strip().title()
    return None

def main():
    ap = argparse.ArgumentParser("Build a compact corpus from data/live/")
    ap.add_argument("--src", type=Path, default=Path("data/live"), help="Folder with utter_*.wav")
    ap.add_argument("--speaker-db", type=Path, default=Path("data/speakers/default.npz"))
    ap.add_argument("--out", type=Path, default=Path("corpus"))
    ap.add_argument("--audio-format", choices=["flac", "wav"], default="flac")
    ap.add_argument("--sr", type=int, default=16000, help="Resample target sample rate")
    ap.add_argument("--channels", type=int, default=1, help="Number of channels in export")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing corpus folder")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    audio_out = args.out / "audio"
    if audio_out.exists() and args.overwrite:
        shutil.rmtree(audio_out)
    audio_out.mkdir(parents=True, exist_ok=True)

    # Copy current speaker DB snapshot into corpus
    spk_db_out = args.out / "speakers_db.npz"
    if args.speaker_db.exists():
        shutil.copyfile(args.speaker_db, spk_db_out)

    # Init SpeakerID in read-write mode (centroids may update while we build).
    # That’s fine; we already copied the snapshot above for the corpus.
    spk = SpeakerID(db_path=args.speaker_db)

    # Collect files
    wavs = sorted(args.src.glob("utter_*.wav"))
    if not wavs:
        print(f"No wavs found in {args.src}")
        return

    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "utt_id",
            "orig_path",
            "export_path",
            "duration_sec",
            "rms",
            "transcript",
            "claimed_name",
            "pred_label",
            "pred_sim"
        ])

        for i, w in enumerate(wavs, 1):
            # Read audio for duration/rms and resave compact
            audio, sr = sf.read(str(w), dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio[:, 0]
            dur = len(audio) / float(sr or 1)
            r = rms(audio)

            # Export compact audio
            utt_id = w.stem  # e.g., "utter_2025-08-15_12-46-06"
            if args.audio_format == "flac":
                out_path = audio_out / f"{utt_id}.flac"
                sf.write(str(out_path), audio, sr, format="FLAC")
            else:
                # resave as 16k mono wav to reduce size a bit (optional)
                out_path = audio_out / f"{utt_id}.wav"
                sf.write(str(out_path), audio, sr, subtype="PCM_16")

            # Transcribe
            text = (transcribe_file(str(w)) or "").strip()

            # Labels: prefer explicit claimed names in transcript; else SpeakerID
            claimed = extract_name(text)
            if claimed:
                # Do not mutate the DB here; we want the model’s prediction independent of claim
                # So we just record the claim as "claimed_name" and still run identify()
                pass

            # Predict label
            try:
                pred_label, pred_sim = spk.identify(w)  # returns (label, similarity)
            except Exception:
                pred_label, pred_sim = ("Unknown", 0.0)

            writer.writerow([
                utt_id,
                str(w),
                str(out_path.relative_to(args.out)),
                f"{dur:.3f}",
                f"{r:.6f}",
                text,
                claimed or "",
                pred_label or "",
                f"{float(pred_sim):.4f}"
            ])

    # Auto README for the corpus
    readme = args.out / "README_corpus.md"
    readme.write_text(
        f"""# Corpus

This folder contains a compact snapshot of the multi-speaker real-time agent runs.

- **audio/**: utterances re-saved as {args.audio_format.upper()} for size efficiency
- **manifest.csv**: one row per utterance with transcript, predicted label, similarity, and basics
- **speakers_db.npz**: snapshot of the speaker embedding centroids at build time (optional)
- Source of audio files: `{args.src}`

## Columns in manifest.csv
- `utt_id`: basename of the utterance (timestamp-based)
- `orig_path`: original path under `{args.src}`
- `export_path`: relative path under `audio/`
- `duration_sec`: float seconds
- `rms`: root-mean-square amplitude of the original audio
- `transcript`: ASR output (Whisper)
- `claimed_name`: name extracted from transcript if the speaker introduced themselves in that utterance
- `pred_label`: speaker label predicted by the SpeakerID model
- `pred_sim`: cosine similarity to the predicted speaker centroid
""",
        encoding="utf-8",
    )

    print(f"\n[OK] Corpus built at: {args.out}")
    print(f"     Audio: {audio_out}")
    print(f"     Manifest: {manifest}")
    print(f"     Speaker DB snapshot: {spk_db_out if spk_db_out.exists() else '(none)'}")

if __name__ == "__main__":
    main()

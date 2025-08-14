# summarize_sessions.py
"""
Scan data/logs/* session folders and build an aggregate CSV:
- session_id, started_at, mode (one-shot | multi_from_file)
- models (whisper/llm/tts), device, python
- timings (acquire/diarize/asr/llm/tts)
- text sizes (transcript_chars, reply_chars)
- context (n_speakers, last_speaker)
- conversation stats for multi_from_file (n_turns, speakers_list)
"""
import csv
import json
import sys
from pathlib import Path

LOGS_DIR = Path("data/logs")

def read_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def read_conversation_index(path: Path):
    """Return (turn_rows, speakers_set) for multi_from_file sessions."""
    if not path.exists():
        return [], set()
    rows = []
    speakers = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
            speakers.add(row.get("speaker", ""))
    return rows, speakers

def detect_mode(session_dir: Path) -> str:
    # Heuristic: multi_from_file sessions are named *_multi_from_file
    if session_dir.name.endswith("_multi_from_file"):
        return "multi_from_file"
    return "one-shot"

def summarize_session(session_dir: Path):
    m = read_json(session_dir / "metrics.json")
    started_at = m.get("started_at", "")
    platform = m.get("platform", {})
    models = m.get("models", {})
    timings = m.get("timings", {})
    meta = m.get("meta", {})
    context = m.get("context", {})

    mode = detect_mode(session_dir)

    # Defaults (safe if fields missing)
    system = platform.get("system", "")
    release = platform.get("release", "")
    python_ver = platform.get("python", "")
    device = platform.get("device", "")

    whisper_model = models.get("whisper", "")
    llm_model = models.get("llm", "")
    tts_model = models.get("tts", "")

    acquire_s = timings.get("acquire_s", "")
    diarize_s = timings.get("diarize_s", "")
    asr_s = timings.get("asr_s", "")
    llm_s = timings.get("llm_s", "")
    tts_s = timings.get("tts_s", "")

    transcript_chars = meta.get("transcript_chars", "")
    reply_chars = meta.get("reply_chars", "")

    participants = context.get("participants", [])
    last_speaker = context.get("last_speaker", "")
    n_speakers = len(participants) if participants else ""

    # Multi conversation extras
    n_turns = ""
    speakers_list = ""
    if mode == "multi_from_file":
        conv_csv = session_dir / "conversation_index.csv"
        turns, spk_set = read_conversation_index(conv_csv)
        n_turns = len(turns)
        speakers_list = ",".join(sorted([s for s in spk_set if s]))

    return {
        "session_id": session_dir.name,
        "mode": mode,
        "started_at": started_at,
        "system": system,
        "os_release": release,
        "python": python_ver,
        "device": device,
        "whisper_model": whisper_model,
        "llm_model": llm_model,
        "tts_model": tts_model,
        "acquire_s": acquire_s,
        "diarize_s": diarize_s,
        "asr_s": asr_s,
        "llm_s": llm_s,
        "tts_s": tts_s,
        "transcript_chars": transcript_chars,
        "reply_chars": reply_chars,
        "n_speakers": n_speakers,
        "last_speaker": last_speaker,
        "n_turns": n_turns,
        "speakers_list": speakers_list,
    }

def main():
    if not LOGS_DIR.exists():
        print("No data/logs directory found.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for session_dir in sorted(LOGS_DIR.iterdir()):
        if not session_dir.is_dir():
            continue
        # Require metrics.json to consider it a session
        if not (session_dir / "metrics.json").exists():
            continue
        rows.append(summarize_session(session_dir))

    if not rows:
        print("No sessions with metrics.json found under data/logs.", file=sys.stderr)
        sys.exit(1)

    out_csv = LOGS_DIR / "sessions_summary.csv"
    fieldnames = [
        "session_id", "mode", "started_at",
        "system", "os_release", "python", "device",
        "whisper_model", "llm_model", "tts_model",
        "acquire_s", "diarize_s", "asr_s", "llm_s", "tts_s",
        "transcript_chars", "reply_chars",
        "n_speakers", "last_speaker",
        "n_turns", "speakers_list",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[summary] Wrote {out_csv} ({len(rows)} sessions)")

if __name__ == "__main__":
    main()

# multi_from_file.py
import csv
import datetime
from pathlib import Path
from typing import List, Tuple, Dict

from config import load_config
from asr.diarize import diarize_wav
from asr.transcribe import transcribe_file
from llm.generate import generate_response, build_speaker_aware_system_prompt
from tts.speak import speak_text
from orchestrator import make_turns_from_diarization, format_tagged_transcript


# ----------------------------
# Lightweight per-speaker memory
# ----------------------------
class LocalSpeakerMemory:
    """Keeps the last N utterances per speaker (very simple)."""
    def __init__(self, max_items: int = 3):
        self.max_items = max_items
        self.buffers: Dict[str, List[str]] = {}

    def add(self, speaker: str, text: str):
        buf = self.buffers.setdefault(speaker, [])
        buf.append((text or "").strip())
        if len(buf) > self.max_items:
            del buf[0 : len(buf) - self.max_items]

    def context_text(self, speaker: str) -> str:
        buf = self.buffers.get(speaker, [])
        if not buf:
            return ""
        joined = " | ".join(buf)
        return f"Recent history for {speaker}: {joined}"


# ----------------------------
# Tiny topic persona helper
# ----------------------------
def topic_persona_hint(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["visa", "flight", "hotel", "travel", "uk", "usa", "canterbury"]):
        return "Be concise and practical about travel info (dates, weather, documents)."
    if any(k in t for k in ["news", "election", "biden", "trump", "policy", "parliament", "government"]):
        return "Be neutral and factual on news/politics; avoid speculation."
    if any(k in t for k in ["api", "python", "model", "gpu", "dataset", "audio", "diarization"]):
        return "Give clear technical help, short and actionable."
    if any(k in t for k in ["health", "symptom", "medicine", "doctor"]):
        return "Give general information only; advise seeing a professional for medical decisions."
    return "Keep replies brief (1–3 sentences). Avoid lists unless asked."


# ----------------------------
# Runner
# ----------------------------
def run(file_path: str):
    cfg = load_config()
    in_wav = Path(file_path)
    if not in_wav.exists():
        raise FileNotFoundError(in_wav)

    # Create a session-like folder (with a suffix so it’s clear this came from file)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = Path("data/logs") / f"{timestamp}_multi_from_file"
    turns_dir = session_dir / "turns"
    session_dir.mkdir(parents=True, exist_ok=True)

    # 1) Diarize entire file (graceful fallback if token/pipeline missing)
    try:
        segments = diarize_wav(str(in_wav))  # -> list of Segment(start,end,speaker)
    except Exception as e:
        print(f"[multi] Diarization failed ({e}). Falling back to single-speaker.")
        class _S:  # tiny shim to match Segment fields
            def __init__(self, start, end, speaker): self.start, self.end, self.speaker = start, end, speaker
        segments = [_S(0.0, 9e9, "spk_00")]

    # Save raw diarization (even if fallback)
    with (session_dir / "diarization.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "speaker"])
        for s in segments:
            w.writerow([round(float(s.start), 3), round(float(s.end), 3), str(s.speaker)])

    participants = sorted({str(s.speaker) for s in segments})
    # last speaker across full file (might equal the only speaker in fallback)
    last_speaker_global = max(segments, key=lambda s: float(s.end)).speaker if segments else None

    # 2) Merge into speaker turns + export per-turn wav clips
    rows: List[Tuple[float, float, str]] = [(float(s.start), float(s.end), str(s.speaker)) for s in segments]
    turns = make_turns_from_diarization(in_wav, rows, turns_dir, gap=0.5, min_turn_duration=0.5)


    if not turns:
        print("[multi] No valid turns found after diarization merge; creating a single full-file turn.")
        # In weird cases, write a single turn using the full file bounds
        turns = make_turns_from_diarization(in_wav, [(0.0, 9e9, "spk_00")], turns_dir)

    # 3) Process each turn -> transcript -> LLM reply (speaker-aware + memory) -> TTS
    base_prompt = cfg["llm"]["base_system_prompt"]
    model_name = cfg["llm"]["model"]
    voice_id = cfg.get("tts", {}).get("voice_id")
    model_id = cfg.get("tts", {}).get("model_id")

    index_rows = []
    memory = LocalSpeakerMemory(max_items=3)

    per_turn_texts: List[str] = []

    for i, turn in enumerate(turns, 1):
        # (a) Transcribe this speaker slice
        transcript = (transcribe_file(str(turn.wav_path)) or "").strip()
        per_turn_texts.append(transcript or "")

        # (b) Build a system prompt that includes:
        #     - base system prompt
        #     - list of participants
        #     - current (last) speaker to address
        #     - short per-speaker recent context
        #     - a topic-sensitive “persona” hint
        persona = topic_persona_hint(transcript)
        sys_core = build_speaker_aware_system_prompt(
            base_prompt=base_prompt,
            participants=participants,
            last_speaker=turn.speaker,  # address the current talker
        )
        history = memory.context_text(turn.speaker)
        system_prompt = "\n".join(p for p in [sys_core, history, persona] if p)

        # (c) LLM reply
        reply = (generate_response(transcript, system_prompt=system_prompt, model_name=model_name) or "").strip()

        # (d) Save artefacts beside each turn
        (session_dir / f"turn_{i:02d}_{turn.speaker}_transcript.txt").write_text(transcript, encoding="utf-8")
        (session_dir / f"turn_{i:02d}_{turn.speaker}_system_prompt.txt").write_text(system_prompt, encoding="utf-8")
        (session_dir / f"turn_{i:02d}_{turn.speaker}_response.txt").write_text(reply, encoding="utf-8")

        # (e) TTS per turn (no auto-play; save for later review)
        tts_mp3 = session_dir / f"turn_{i:02d}_{turn.speaker}_response.mp3"
        speak_text(reply, save_to=tts_mp3, play=False, voice_id=voice_id, model_id=model_id)

        # (f) Update memory AFTER replying (so next time this speaker talks we have context)
        memory.add(turn.speaker, transcript)

        # For the global index table
        index_rows.append([
            i,
            turn.speaker,
            round(turn.start, 3),
            round(turn.end, 3),
            transcript.replace("\n", " "),
            reply.replace("\n", " "),
            str(tts_mp3.name),
        ])

    # 4) Save a conversation index CSV (easy to skim)
    with (session_dir / "conversation_index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["turn", "speaker", "start_s", "end_s", "transcript", "reply", "tts_file"])
        w.writerows(index_rows)

    # 5) Save a speaker-tagged transcript for quick reading
    try:
        tagged = format_tagged_transcript(turns, per_turn_texts, show_timestamps=True)
        (session_dir / "speaker_transcript.txt").write_text(tagged, encoding="utf-8")
    except Exception as e:
        print(f"[multi] Could not write speaker_transcript.txt: {e}")

    print(f"[multi] Done. Folder: {session_dir}")


if __name__ == "__main__":
    # Usage:
    #   python multi_from_file.py data/test/podcast.wav
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("file", help="Path to a multi-speaker WAV/MP3 file")
    args = p.parse_args()
    run(args.file)

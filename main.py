# main.py
import argparse
import datetime
import logging
import sys
import time
import json
import platform
from pathlib import Path
from typing import Optional

import soundfile as sf
from config import load_config
from asr.transcribe import record_audio, transcribe_audio
from llm.generate import generate_response, build_speaker_aware_system_prompt
from tts.speak import speak_text

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger("speech-agent")


def configure_logging(session_dir: Path):
    """Log to console and to session-specific file."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(session_dir / "run.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def safe_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def run_session(
    use_existing_audio: Optional[str],
    run_diarization: bool,
    duration: int,
    target_sr: int = 16000,
    play_audio: bool = True,
):
    """Main pipeline for one session."""
    cfg = load_config()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = Path("data/logs") / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(session_dir)
    logger.info("Session start")

    # Metrics dict
    device = "cuda" if (torch and hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
    metrics = {
        "started_at": timestamp,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
            "device": device,
        },
        "models": {
            "whisper": cfg.get("asr", {}).get("whisper_model", "small"),
            "llm": cfg.get("llm", {}).get("model", "gpt-3.5-turbo"),
            "tts": f"{cfg.get('tts', {}).get('voice_id','')}/{cfg.get('tts', {}).get('model_id','')}",
        },
        "timings": {},
    }

    try:
        # 1) Acquire audio
        t0 = time.perf_counter()
        user_wav = session_dir / "user.wav"

        if use_existing_audio:
            in_path = Path(use_existing_audio)
            if not in_path.exists():
                raise FileNotFoundError(f"Audio file not found: {in_path}")

            audio, sr = sf.read(str(in_path), dtype="float32", always_2d=False)
            if hasattr(audio, "ndim") and audio.ndim == 2:
                audio = audio[:, 0]  # take first channel if stereo
            sf.write(user_wav, audio, sr)
            logger.info(f"Loaded audio from {in_path}, saved -> {user_wav}")
        else:
            logger.info(f"Recording {duration}s at {target_sr} Hz…")
            audio = record_audio(duration=duration, samplerate=target_sr)
            sf.write(user_wav, audio, target_sr)
            sr = target_sr
            logger.info(f"Saved microphone audio -> {user_wav}")

        metrics["timings"]["acquire_s"] = round(time.perf_counter() - t0, 3)

        # 2) Optional diarization
        participants, last_speaker = [], None
        if run_diarization:
            t0 = time.perf_counter()
            try:
                from asr.diarize import diarize_wav, save_segments_csv

                segments = diarize_wav(str(user_wav))
                save_segments_csv(segments, str(session_dir / "diarization.csv"))
                spk_set = {s.speaker for s in segments}
                participants = sorted(spk_set)
                if segments:
                    last_speaker = max(segments, key=lambda s: s.end).speaker
                logger.info(
                    f"Diarization: {len(segments)} turns, ~{len(spk_set)} speakers, last={last_speaker}"
                )
            except Exception as e:
                logger.warning(f"Diarization skipped: {e}")
            metrics["timings"]["diarize_s"] = round(time.perf_counter() - t0, 3)

        # 3) Transcribe
        t0 = time.perf_counter()
        logger.info("Transcribing…")
        whisper_model = cfg.get("asr", {}).get("whisper_model", "small")
        try:
            # Prefer new signature that accepts model_name
            transcript = transcribe_audio(audio, samplerate=sr, model_name=whisper_model)  # type: ignore
        except TypeError:
            # Backward compatibility with older transcribe_audio(audio, samplerate)
            transcript = transcribe_audio(audio, samplerate=sr)
        safe_write_text(session_dir / "transcript.txt", transcript)
        metrics["timings"]["asr_s"] = round(time.perf_counter() - t0, 3)

        # record transcript length
        metrics.setdefault("meta", {})["transcript_chars"] = len(transcript or "")

        # 4) LLM reply (speaker-aware)
        t0 = time.perf_counter()
        logger.info("Generating reply…")
        base_prompt = cfg.get("llm", {}).get("base_system_prompt", "")
        system_prompt = build_speaker_aware_system_prompt(
            base_prompt=base_prompt,
            participants=participants,
            last_speaker=last_speaker,
        )

        # Save the exact system prompt used (transparency/debugging)
        safe_write_text(session_dir / "system_prompt.txt", system_prompt)

        # Add diarization context into metrics
        metrics.setdefault("context", {})["participants"] = participants
        metrics["context"]["last_speaker"] = last_speaker

        reply = generate_response(
            transcript,
            system_prompt=system_prompt,
            model_name=cfg.get("llm", {}).get("model", "gpt-3.5-turbo"),
        )
        safe_write_text(session_dir / "response.txt", reply)
        metrics["timings"]["llm_s"] = round(time.perf_counter() - t0, 3)

        # record reply length
        metrics.setdefault("meta", {})["reply_chars"] = len((reply or "").strip())

        # Also print to console for quick feedback
        print("\n=== Agent Reply ===")
        print((reply or "").strip() or "[empty reply]")
        print("===================\n")

        # 5) TTS (save MP3 to session dir)
        t0 = time.perf_counter()
        logger.info("Speaking…")
        tts_mp3 = session_dir / "response.mp3"
        try:
            speak_text(
                reply or "",
                save_to=tts_mp3,
                play=bool(play_audio),
                voice_id=cfg.get("tts", {}).get("voice_id"),
                model_id=cfg.get("tts", {}).get("model_id"),
            )
            logger.info(f"TTS saved -> {tts_mp3}")
        except Exception as e:
            logger.warning(f"TTS skipped: {e}")
        metrics["timings"]["tts_s"] = round(time.perf_counter() - t0, 3)

        # 6) Append summary log
        (Path("data/logs")).mkdir(parents=True, exist_ok=True)
        with open("data/logs/session_summary.log", "a", encoding="utf-8") as f:
            f.write(f"{timestamp} | USER: {transcript} | AGENT: {reply}\n")

        # Save metrics.json
        (session_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        logger.info(f"Metrics saved -> {session_dir / 'metrics.json'}")

        logger.info("Session complete ✅")

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        raise


def parse_args():
    p = argparse.ArgumentParser(description="Speech Agent session runner")
    p.add_argument("--file", help="Process an existing audio file instead of recording (wav/mp3)")
    p.add_argument("--no-diarize", action="store_true", help="Disable speaker diarization")
    p.add_argument("--no-play", action="store_true", help="Do not play audio aloud (still saves file)")
    p.add_argument("--duration", type=int, default=7, help="Recording duration (seconds)")
    p.add_argument("--samplerate", type=int, default=16000, help="Recording sample rate (Hz)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config()

    # CLI args take precedence over config
    duration = args.duration if args.duration else cfg["app"]["duration"]
    samplerate = args.samplerate if args.samplerate else cfg["app"]["samplerate"]

    run_session(
        use_existing_audio=args.file,
        run_diarization=(not args.no_diarize) and cfg["app"]["diarization"],
        duration=duration,
        target_sr=samplerate,
        play_audio=not args.no_play,
    )

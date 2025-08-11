# main.py
import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Optional

import soundfile as sf
from config import load_config
from asr.transcribe import record_audio, transcribe_audio
from llm.generate import generate_response, build_speaker_aware_system_prompt
from tts.speak import speak_text

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
    path.write_text(text, encoding="utf-8")


def run_session(
    use_existing_audio: Optional[str],
    run_diarization: bool,
    duration: int,
    target_sr: int = 16000
):
    """Main pipeline for one session."""
    cfg = load_config()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = Path("data/logs") / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(session_dir)
    logger.info("Session start")

    try:
        # 1) Acquire audio
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

        # 2) Optional diarization
        participants, last_speaker = [], None
        if run_diarization:
            try:
                from asr.diarize import diarize_wav, save_segments_csv
                segments = diarize_wav(str(user_wav))
                save_segments_csv(segments, str(session_dir / "diarization.csv"))
                spk_set = {s.speaker for s in segments}
                participants = sorted(spk_set)
                if segments:
                    last_speaker = max(segments, key=lambda s: s.end).speaker
                logger.info(f"Diarization: {len(segments)} turns, ~{len(spk_set)} speakers, last={last_speaker}")
            except Exception as e:
                logger.warning(f"Diarization skipped: {e}")

        # 3) Transcribe
        logger.info("Transcribing…")
        transcript = transcribe_audio(audio, samplerate=sr)
        safe_write_text(session_dir / "transcript.txt", transcript)

        # 4) LLM reply (speaker-aware)
        logger.info("Generating reply…")
        base_prompt = cfg["llm"]["base_system_prompt"]
        system_prompt = build_speaker_aware_system_prompt(
            base_prompt=base_prompt,
            participants=participants,
            last_speaker=last_speaker,
        )
        reply = generate_response(
            transcript,
            system_prompt=system_prompt,
            model_name=cfg["llm"]["model"]
        )
        safe_write_text(session_dir / "response.txt", reply)

        # 5) TTS (save MP3 to session dir)
        logger.info("Speaking…")
        tts_mp3 = session_dir / "response.mp3"
        speak_text(reply, save_to=tts_mp3)
        logger.info(f"TTS saved -> {tts_mp3}")

        # 6) Append summary log
        with open("data/logs/session_summary.log", "a", encoding="utf-8") as f:
            f.write(f"{timestamp} | USER: {transcript} | AGENT: {reply}\n")

        logger.info("Session complete ✅")

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        raise


def parse_args():
    p = argparse.ArgumentParser(description="Speech Agent session runner")
    p.add_argument("--file", help="Process an existing audio file instead of recording (wav/mp3)")
    p.add_argument("--no-diarize", action="store_true", help="Disable speaker diarization")
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
    )

"""
Real-time speech agent (non-blocking):
- InputStream with callback -> Queue (no blocking in calibration or loop)
- Simple RMS VAD (frame_ms, end_silence_ms)
- Optional speaker labeling (--spk-id) via utils.speaker_id (Resemblyzer/MFCC)
- Per-speaker short-term memory (utils.memory.SpeakerMemory)
- Transcribe -> LLM -> (optional) TTS
- Speaker DB persistence: --speaker-db, --reset-speakers, --list-speakers
"""

import argparse
import queue
import re
import signal
import time
import wave
from pathlib import Path
from typing import List, Optional

import numpy as np
import sounddevice as sd

from config import load_config
from asr.transcribe import transcribe_file
from llm.generate import generate_response
from tts.speak import speak_text
from utils.memory import SpeakerMemory

try:
    from utils.speaker_id import SpeakerID
except Exception:
    SpeakerID = None  # type: ignore

# Defaults
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_MS = 30
DEFAULT_END_SILENCE_MS = 700
DEFAULT_MAX_UTTERANCE_SEC = 30
DEFAULT_CALIBRATION_SEC = 0.8
DEFAULT_RMS_SILENCE_MULT = 2.5

SESSION_DIR = Path("data/live")
SESSION_DIR.mkdir(parents=True, exist_ok=True)

_running = True


def _graceful_exit(sig, frame):
    global _running
    _running = False


signal.signal(signal.SIGINT, _graceful_exit)
signal.signal(signal.SIGTERM, _graceful_exit)


def _flush(msg: str):
    print(msg, flush=True)


def write_wav(path: Path, int16_audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sr)
        wf.writeframes(int16_audio.tobytes())


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x.astype(np.float32)))))


def _extract_name_from_text(text: str) -> Optional[str]:
    t = (text or "").strip()
    m = re.search(r"\b(i am|i'm|this is|my name is|call me)\s+([A-Za-z][A-Za-z\-']{1,30})\b", t, flags=re.I)
    if m:
        return m.group(2).strip().title()
    return None


def _unique_ints(seq):
    out, seen = [], set()
    for x in seq:
        if x is None:
            continue
        xi = int(x)
        if xi in seen:
            continue
        seen.add(xi)
        out.append(xi)
    return out


def _open_stream_with_callback(device: Optional[int], preferred_sr: int, frame_ms: int):
    q = queue.Queue(maxsize=64)

    def make_cb(channels: int):
        def _cb(indata, frames, time_info, status):
            try:
                x = indata[:, 0] if indata.ndim > 1 else indata
                q.put_nowait(x.astype(np.float32, copy=False).copy())
            except queue.Full:
                pass
        return _cb

    dev_info = None
    if device is not None:
        try:
            dev_info = sd.query_devices(device)
            _flush(
                f"[Live] Using device {device}: {dev_info['name']} "
                f"(in={dev_info['max_input_channels']}, default_sr={dev_info.get('default_samplerate')})"
            )
            sd.default.device = (device, None)
        except Exception as e:
            _flush(f"[Live] Could not query device {device}: {e}")

    default_sr = int(dev_info["default_samplerate"]) if (dev_info and dev_info.get("default_samplerate")) else None
    sr_candidates = _unique_ints([preferred_sr, default_sr, 48000, 44100, 16000])

    attempt_cfgs = [
        dict(dtype="float32", channels=1),
        dict(dtype="float32", channels=2),
        dict(dtype="int16", channels=1),
        dict(dtype="int16", channels=2),
    ]

    last_err = None
    for sr in sr_candidates:
        frame_samples = max(1, int(sr * frame_ms / 1000))
        for cfg in attempt_cfgs:
            try:
                sd.check_input_settings(
                    device=device if device is not None else None,
                    samplerate=sr,
                    channels=cfg["channels"],
                    dtype=cfg["dtype"],
                )
            except Exception as e:
                last_err = e
                continue
            try:
                stream = sd.InputStream(
                    samplerate=sr,
                    device=device if device is not None else None,
                    blocksize=0,
                    dtype=cfg["dtype"],
                    channels=cfg["channels"],
                    callback=make_cb(cfg["channels"]),
                )
                stream.start()
                _flush(f"[Live] Stream started at {sr} Hz with {cfg}")
                return stream, q, sr, frame_samples
            except Exception as e:
                last_err = e
                continue

    raise RuntimeError(f"Could not open input stream (device={device}). Last error: {last_err}")


def _gather(q: "queue.Queue[np.ndarray]", needed: int, timeout_s: float) -> np.ndarray:
    chunks: List[np.ndarray] = []
    got, t0 = 0, time.time()
    while got < needed and (time.time() - t0) < timeout_s and _running:
        try:
            x = q.get(timeout=0.2)
            chunks.append(x)
            got += len(x)
        except queue.Empty:
            pass
    if not chunks:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _safe_label_from_enroll(spk, claimed: str, wav_path: Path) -> str:
    try:
        ret = spk.enroll(claimed, wav_path)
        if isinstance(ret, str) and ret.strip():
            return ret.strip().title()
        return claimed.title()
    except Exception:
        return claimed.title()


def realtime_loop(
    device: Optional[int],
    samplerate: int,
    frame_ms: int,
    end_silence_ms: int,
    max_utterance_sec: int,
    calibration_sec: float,
    rms_silence_mult: float,
    play_audio: bool,
    use_spkid: bool,
    speaker_db: Optional[Path],
    list_only: bool,
    reset_only: bool,
    spk_threshold: float,
    spk_margin: float,
    mem_turns: int,
) -> None:
    cfg = load_config()

    # Per-speaker memory
    memory = SpeakerMemory(max_turns=mem_turns)

    # Speaker-ID (with persistence wired correctly)
    spk = None
    if use_spkid:
        if SpeakerID is None:
            _flush("[ID] SpeakerID unavailable; running without")
            use_spkid = False
        else:
            _flush("[ID] Initializing SpeakerID backend…")
            spk = SpeakerID(
                sample_rate=16000,
                threshold=spk_threshold,
                margin=spk_margin,
                db_path=speaker_db,         # << ensure save/load go to the same file
            )
            _flush(f"[ID] SpeakerID backend: {getattr(spk, 'mode', 'unknown')}")

            # list/reset modes happen before we open the stream
            if list_only:
                spk.load()  # load current DB
                labs = spk.labels()
                if labs:
                    _flush("[ID] Speakers:")
                    for l in labs:
                        _flush(f"  - {l}")
                else:
                    _flush("[ID] No speakers in DB.")
                return

            if reset_only:
                spk.reset()
                spk.save()
                _flush("[ID] Speaker DB reset.")
                return

    _flush("[Live] Opening stream…")
    stream, q, used_sr, frame_samples = _open_stream_with_callback(device, samplerate, frame_ms)

    # Calibrate (non-blocking)
    _flush("[Live] Calibrating noise floor…")
    need_calib = max(1, int(used_sr * calibration_sec))
    calib = _gather(q, needed=need_calib, timeout_s=calibration_sec + 0.8)
    noise_rms = rms(calib) if calib.size else 0.0004
    silence_thresh = max(0.0004, noise_rms * rms_silence_mult)
    _flush(f"[Live] Noise RMS≈{noise_rms:.4f} → silence threshold≈{silence_thresh:.4f}")

    end_silence_frames = max(1, end_silence_ms // frame_ms)
    in_speech = False
    utter_frames: List[np.ndarray] = []
    silence_run = 0
    utter_start_ts: Optional[float] = None

    _flush("[Live] Listening… (Ctrl+C to stop)")
    while _running:
        frame = _gather(q, needed=frame_samples, timeout_s=0.8)
        if frame.size == 0:
            continue

        frame_rms = rms(frame)

        if frame_rms > silence_thresh:
            if not in_speech:
                in_speech = True
                utter_frames = []
                utter_start_ts = time.time()
            utter_frames.append(frame.copy())
            silence_run = 0
            if (time.time() - (utter_start_ts or time.time())) > max_utterance_sec:
                in_speech = False
        else:
            if in_speech:
                silence_run += 1
                utter_frames.append(frame.copy())
                if silence_run >= end_silence_frames:
                    in_speech = False

        # Process utterance
        if (not in_speech) and utter_frames:
            ts = time.strftime("%Y-%m-%d_%H-%M-%S")
            audio_f32 = np.concatenate(utter_frames).astype(np.float32)
            audio_i16 = np.clip(audio_f32 * 32767.0, -32768.0, 32767.0).astype(np.int16)
            wav_path = SESSION_DIR / f"utter_{ts}.wav"
            write_wav(wav_path, audio_i16, used_sr)
            _flush(f"[Live] Utterance -> {wav_path}")

            # Transcribe
            _flush("[Live] Transcribing…")
            text = (transcribe_file(str(wav_path)) or "").strip()
            _flush(f"[Live] You said: {text or '[empty]'}")

            # Speaker label (opt-in)
            label: Optional[str] = None
            sim = 0.0
            if use_spkid and spk is not None:
                claimed = _extract_name_from_text(text)
                if claimed:
                    label = _safe_label_from_enroll(spk, claimed, wav_path)
                    sim = 1.0
                    spk.save()  # persist add/update
                else:
                    lab, s = spk.identify(wav_path)
                    label, sim = (lab, float(s))
                    spk.save()  # persist centroid updates / guest minting
                _flush(f"[ID] Labeled: {label or 'Unknown'} (sim={sim:.2f})")

            # LLM (speaker-aware with per-speaker memory)
            _flush("[Live] Thinking…")
            base_prompt = cfg.get("llm", {}).get(
                "base_system_prompt",
                "You are a concise, helpful multi-party assistant. Keep replies to 1–3 sentences. "
                "Keep each thread separate; do not mix details across speakers.",
            )
            model_name = cfg.get("llm", {}).get("model", "gpt-3.5-turbo")

            if label:
                history = memory.format_for_prompt(label)
                if history and history != "None":
                    system_prompt = (
                        base_prompt.strip()
                        + f"\nAddress the person named {label} directly."
                        + f"\nRecent context for {label}:\n{history}"
                    )
                else:
                    system_prompt = base_prompt.strip() + f"\nAddress the person named {label} directly."
            else:
                system_prompt = base_prompt

            reply = (generate_response(text, system_prompt=system_prompt, model_name=model_name) or "").strip()
            if label and not reply.lower().startswith(label.lower()):
                reply = f"{label}, {reply}"

            _flush(f"[Live] Agent{(' → ' + label) if label else ''}: {reply or '[empty]'}")

            # Persist reply text (useful for corpus)
            try:
                (SESSION_DIR / f"utter_{ts}_reply.txt").write_text(reply or "", encoding="utf-8")
            except Exception:
                pass

            # Update per-speaker memory AFTER replying
            memory.add(label or "Unknown", user_text=text, agent_text=reply or "")

            # TTS (pause stream to avoid feedback)
            stream.stop()
            try:
                voice_id = cfg.get("tts", {}).get("voice_id")
                model_id = cfg.get("tts", {}).get("model_id")
                speak_text(
                    reply,
                    save_to=SESSION_DIR / f"utter_{ts}_reply.mp3",
                    play=play_audio,
                    voice_id=voice_id,
                    model_id=model_id,
                )
                res = speak_text(
                    reply,
                    save_to=SESSION_DIR / f"utter_{ts}_reply.mp3",
                    play=play_audio,
                    voice_id=voice_id,
                    model_id=model_id,
                )
                if res and res.get("note"):
                    _flush(f"[TTS] {res['note']}")
            except Exception as e:
                _flush(f"[Live] TTS error (continuing): {e}")
            finally:
                try:
                    _ = _gather(q, needed=frame_samples * 2, timeout_s=0.2)
                    stream.start()
                except Exception as e:
                    _flush(f"[Live] Could not restart stream: {e}")
                    break

            # Reset
            utter_frames = []
            silence_run = 0
            utter_start_ts = None

    try:
        stream.stop()
        stream.close()
    except Exception:
        pass
    _flush("\n[Live] Stopped.")


def parse_args():
    p = argparse.ArgumentParser(description="Real-time speech agent (non-blocking Queue + RMS VAD)")
    p.add_argument("--device", type=int, default=None, help="Input device index (see `python -m sounddevice`)")
    p.add_argument("--samplerate", type=int, default=DEFAULT_SAMPLE_RATE, help="Preferred sample rate (Hz)")
    p.add_argument("--frame-ms", type=int, default=DEFAULT_FRAME_MS, help="Analysis frame size (ms)")
    p.add_argument("--end-silence-ms", type=int, default=DEFAULT_END_SILENCE_MS, help="Silence to end utterance (ms)")
    p.add_argument("--max-utterance-sec", type=int, default=DEFAULT_MAX_UTTERANCE_SEC, help="Max utterance length (sec)")
    p.add_argument("--calibration-sec", type=float, default=DEFAULT_CALIBRATION_SEC, help="Calibration duration (sec)")
    p.add_argument("--no-play", action="store_true", help="Do not play audio aloud (still saves reply mp3)")
    p.add_argument("--spk-id", action="store_true", help="Enable speaker labeling via embeddings")
    # Persistence controls
    p.add_argument("--speaker-db", type=Path, default=Path("data/speakers/default.npz"), help="Path to speaker DB (NPZ)")
    p.add_argument("--list-speakers", action="store_true", help="List speakers in DB and exit")
    p.add_argument("--reset-speakers", action="store_true", help="Reset/clear the speaker DB and exit")
    # Speaker-ID tuning + memory
    p.add_argument("--spk-threshold", type=float, default=0.82, help="Similarity threshold for a match")
    p.add_argument("--spk-margin", type=float, default=0.08, help="Temporal smoothing margin below threshold")
    p.add_argument("--mem-turns", type=int, default=3, help="How many recent turns to keep per speaker")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        realtime_loop(
            device=args.device,
            samplerate=args.samplerate,
            frame_ms=args.frame_ms,
            end_silence_ms=args.end_silence_ms,
            max_utterance_sec=args.max_utterance_sec,
            calibration_sec=args.calibration_sec,
            rms_silence_mult=DEFAULT_RMS_SILENCE_MULT,
            play_audio=not args.no_play,
            use_spkid=bool(args.spk_id),
            speaker_db=args.speaker_db,
            list_only=bool(args.list_speakers),
            reset_only=bool(args.reset_speakers),
            spk_threshold=args.spk_threshold,
            spk_margin=args.spk_margin,
            mem_turns=args.mem_turns,
        )
    except KeyboardInterrupt:
        print("\n[Live] Exiting.")

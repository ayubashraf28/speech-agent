"""
Real-time speech agent (non-blocking):
- InputStream with callback -> Queue (no blocking in calibration or loop)
- Simple RMS VAD (frame_ms, end_silence_ms)
- Optional speaker labeling (--spk-id) via utils.speaker_id (Resemblyzer)
- Transcribe -> LLM -> (optional) TTS
"""

import argparse
import queue
import re
import signal
import time
import wave
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import sounddevice as sd

from config import load_config
from asr.transcribe import transcribe_file
from llm.generate import generate_response
from tts.speak import speak_text

# Optional speaker-ID
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


# -------- Safer name-claim extractor (prevents "I'm looking..." → "Looking") --------
_NAME_STOPWORDS = {
    "looking", "going", "doing", "working", "writing", "planning", "thinking", "asking",
    "needing", "nothing", "something", "anything", "everything", "morning", "evening",
    "afternoon", "okay", "ok", "fine", "good", "hello", "hi", "hey", "thanks",
    "thankyou", "thank", "please"
}

def _extract_claimed_name(text: str) -> Tuple[Optional[str], bool]:
    """
    Returns (name, strong_intent).
    strong_intent=True only for 'my name is' / 'call me'.
    For 'I'm'/'I am'/'This is', accept only if it looks like a proper name.
    """
    t = (text or "").strip()

    # Strong intent -> allowed to create a new label
    m = re.search(r"\b(?:my name is|call me)\s+([A-Za-z][A-Za-z\-']{1,30})\b", t, flags=re.I)
    if m:
        tok = m.group(1).strip()
        low = tok.lower()
        if low not in _NAME_STOPWORDS and not low.endswith("ing"):
            return tok.title(), True

    # Soft intent -> only accept if it *looks* like a proper name (Title-case helps)
    m = re.search(r"\b(?:i am|i'm|this is)\s+([A-Za-z][A-Za-z\-']{1,30})\b", t, flags=re.I)
    if m:
        tok = m.group(1).strip()
        low = tok.lower()
        if low not in _NAME_STOPWORDS and not low.endswith("ing") and tok[:1].isalpha():
            if tok[:1].isupper():
                return tok.title(), False

    return None, False
# -------------------------------------------------------------------------------


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
    """
    Try a few (samplerate, dtype, channels) combos.
    Use a callback that pushes float32 mono frames into a Queue (non-blocking).
    """
    q = queue.Queue(maxsize=64)

    def make_cb(channels: int):
        def _cb(indata, frames, time_info, status):
            if status:
                # drop status prints to keep console clean
                pass
            try:
                x = indata[:, 0] if indata.ndim > 1 else indata
                q.put_nowait(x.astype(np.float32, copy=False).copy())
            except queue.Full:
                # drop if we're behind
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
                    blocksize=0,  # let backend choose
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
    """
    Non-blocking gather: pull samples from the queue until we have 'needed'
    or timeout. Returns float32 mono.
    """
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
) -> None:
    cfg = load_config()

    _flush("[Live] Opening stream…")
    stream, q, used_sr, frame_samples = _open_stream_with_callback(device, samplerate, frame_ms)

    # Calibrate (non-blocking)
    _flush("[Live] Calibrating noise floor…")
    need_calib = max(1, int(used_sr * calibration_sec))
    calib = _gather(q, needed=need_calib, timeout_s=calibration_sec + 0.8)
    noise_rms = rms(calib) if calib.size else 0.0004  # tiny absolute fallback
    silence_thresh = max(0.0004, noise_rms * rms_silence_mult)
    _flush(f"[Live] Noise RMS≈{noise_rms:.4f} → silence threshold≈{silence_thresh:.4f}")

    # Optional speaker-ID
    spk = None
    if use_spkid:
        if SpeakerID is None:
            _flush("[ID] SpeakerID unavailable; running without")
            use_spkid = False
        else:
            # Threshold tuned low enough to reuse voices (you can adjust to 0.55–0.70)
            spk = SpeakerID(threshold=0.60, margin=0.08, sample_rate=16000)
            _flush(f"[ID] SpeakerID backend: {spk.mode}")

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

            # Speaker label (opt-in) — robust flow:
            # 1) If claim exists, we try to map to an existing label; otherwise, allow strong-claim enroll.
            # 2) If no claim, try identify; if <threshold, auto-enroll Guest-N.
            label = None
            if use_spkid and spk is not None:
                claimed, strong = _extract_claimed_name(text)

                if claimed:
                    # If an existing label has a close textual name, keep it tidy
                    maybe = spk.best_label_for_name(claimed, min_ratio=0.88 if strong else 0.90)
                    if maybe and maybe != claimed.title():
                        spk.rename(maybe, claimed.title())

                    # Try voice match first
                    id_label, sim = spk.identify(wav_path)
                    if id_label is not None:
                        label = id_label
                        spk.add_sample_to(label, wav_path)  # tighten centroid
                        # If strong claim and label is a Guest, rename to claimed
                        if strong and label.startswith("Guest-"):
                            spk.rename(label, claimed.title())
                            label = claimed.title()
                    else:
                        # No good voice match
                        if strong:
                            spk.enroll(claimed, wav_path)
                            label = claimed.title()
                        else:
                            # soft claim, no good voice → treat as unknown voice
                            label = spk.auto_enroll_guest(wav_path)
                else:
                    # No claim → pure voice ID
                    id_label, sim = spk.identify(wav_path)
                    if id_label is not None:
                        label = id_label
                        spk.add_sample_to(label, wav_path)
                    else:
                        label = spk.auto_enroll_guest(wav_path)

                _flush(f"[ID] Labeled: {label or 'Unknown'}")

            # LLM (speaker-aware if we have a label)
            _flush("[Live] Thinking…")
            base_prompt = cfg.get("llm", {}).get(
                "base_system_prompt",
                "You are a helpful, concise speech agent. Keep answers under 2 sentences.",
            )
            model_name = cfg.get("llm", {}).get("model", "gpt-3.5-turbo")

            system_prompt = base_prompt
            if label:
                system_prompt = (
                    base_prompt.strip()
                    + f"\nAddress the person named {label} directly. "
                      "Do not speculate about other participants."
                )

            reply = (generate_response(text, system_prompt=system_prompt, model_name=model_name) or "").strip()
            if label and not reply.lower().startswith(label.lower()):
                reply = f"{label}, {reply}"

            _flush(f"[Live] Agent{(' → ' + label) if label else ''}: {reply or '[empty]'}")

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
            except Exception as e:
                _flush(f"[Live] TTS error (continuing): {e}")
            finally:
                try:
                    # drain any backlog and resume
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
    p.add_argument("--spk-id", action="store_true", help="Enable speaker labeling via embeddings (Resemblyzer)")
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
        )
    except KeyboardInterrupt:
        print("\n[Live] Exiting.")

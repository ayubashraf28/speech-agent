# realtime_agent.py
"""
Realtime multi-speaker assistant:
- Robust RMS VAD (callback + Queue)
- Speaker labelling via utils.speaker_id (safe: torchaudio or fallback)
- Per-speaker memory (utils.memory.SpeakerMemory)
- Speaker-aware prompts; addresses the addressee by name
"""

import argparse
import collections
import queue
import re
import signal
import sys
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

from config import load_config
from utils.memory import SpeakerMemory
from utils.speaker_id import SpeakerID
from llm.generate import generate_response, build_speaker_aware_system_prompt
from asr.transcribe import transcribe_file
from tts.speak import speak_text

# -------- Tuning defaults --------
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_MS = 20
DEFAULT_END_SILENCE_MS = 1200
DEFAULT_MAX_UTTERANCE_SEC = 30
DEFAULT_CALIBRATION_SEC = 0.8
DEFAULT_ENTER_MULT = 2.0
DEFAULT_EXIT_MULT = 1.4
DEFAULT_RMS_SILENCE_FLOOR = 1e-3
DEFAULT_CALIB_FALLBACK = 0.02
DEFAULT_PREROLL_MS = 180
DEFAULT_MIN_UTTER_MS = 500
DEFAULT_DEBOUNCE_MS = 250

SESSION_DIR = Path("data/live")
SESSION_DIR.mkdir(parents=True, exist_ok=True)

_running = True
_last_seen = {}   # label -> last timestamp seen
_last_label = None

def _graceful_exit(sig, frame):
    global _running
    _running = False

signal.signal(signal.SIGINT, _graceful_exit)
signal.signal(signal.SIGTERM, _graceful_exit)

def _flush_print(msg):
    print(msg)
    sys.stdout.flush()

def write_wav(path, int16_audio, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_audio.tobytes())

def rms(x):
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x.astype(np.float32)))))

def _unique_ints(seq):
    out, seen = [], set()
    for x in seq:
        if x is None:
            continue
        xi = int(x)
        if xi not in seen:
            out.append(xi)
            seen.add(xi)
    return out

def _extract_name_from_text(text: str) -> Optional[str]:
    t = (text or "").strip()
    m = re.search(r"\b(i am|i'm|this is|my name is|call me)\s+([A-Za-z][A-Za-z\-']{1,30})\b", t, flags=re.I)
    if m:
        name = m.group(2).strip().title()
        return name
    return None

def _open_stream_with_callback(device, preferred_sr, frame_ms):
    q = queue.Queue(maxsize=64)

    def make_callback(channels):
        def _cb(indata, frames, t, status):
            if status:
                # Drop status logs to keep console clean
                pass
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
            _flush_print("[Live] Using device %s: %s (in=%s, default_sr=%s)" %
                         (device, dev_info['name'], dev_info['max_input_channels'], dev_info.get('default_samplerate')))
            sd.default.device = (device, None)
        except Exception as e:
            _flush_print("[Live] Could not query device %s: %s" % (device, e))

    default_sr = int(dev_info["default_samplerate"]) if (dev_info and dev_info.get("default_samplerate")) else None
    sr_candidates = _unique_ints([preferred_sr, default_sr, 48000, 44100, 16000])

    attempt_cfgs = [
        dict(dtype="float32", channels=1),
        dict(dtype="float32", channels=2),
        dict(dtype="int16",   channels=1),
        dict(dtype="int16",   channels=2),
    ]

    last_err = None
    for sr in sr_candidates:
        for cfg in attempt_cfgs:
            try:
                sd.check_input_settings(device=device if device is not None else None,
                                        samplerate=sr, channels=cfg["channels"], dtype=cfg["dtype"])
            except Exception as e:
                last_err = e
                continue
            try:
                stream = sd.InputStream(samplerate=sr,
                                        device=device if device is not None else None,
                                        blocksize=0,
                                        dtype=cfg["dtype"],
                                        channels=cfg["channels"],
                                        callback=make_callback(cfg["channels"]))
                stream.start()
                _flush_print("[Live] Stream started at %d Hz with %s" % (sr, cfg))
                return stream, q, sr
            except Exception as e:
                last_err = e
                continue

    raise RuntimeError("Could not open input stream (device=%s). Last error: %s" % (device, last_err))

def _drain_queue(q):
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass

def _gather_samples(q, needed, timeout_s):
    chunks, got, t0 = [], 0, time.time()
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

def realtime_loop(device, samplerate, frame_ms, end_silence_ms, max_utterance_sec,
                  calibration_sec, enter_mult, exit_mult, play_audio,
                  preroll_ms, min_utter_ms, debounce_ms):
    global _last_label  # we assign to this name in the loop

    cfg = load_config()

    # Init speaker ID + memory (safe)
    spk_id = SpeakerID(sample_rate=16000, n_mfcc=40, threshold=0.68, margin=0.08)
    _flush_print(f"[Live] Speaker-ID mode: {spk_id.mode}")
    memory = SpeakerMemory(max_turns=3)

    _flush_print("[Live] Opening stream…")
    stream, q, used_sr = _open_stream_with_callback(device, samplerate, frame_ms)

    # Calibration
    _flush_print("[Live] Calibrating noise floor…")
    n_calib = max(1, int(used_sr * calibration_sec))
    calib_audio = _gather_samples(q, needed=n_calib, timeout_s=calibration_sec + 0.8)
    noise_r = rms(calib_audio) if calib_audio.size else DEFAULT_CALIB_FALLBACK

    enter_thresh = max(DEFAULT_RMS_SILENCE_FLOOR, noise_r * float(enter_mult))
    exit_thresh  = max(DEFAULT_RMS_SILENCE_FLOOR, noise_r * float(exit_mult))
    _flush_print("[Live] Noise RMS≈%.4f → enter≈%.4f, exit≈%.4f" % (noise_r, enter_thresh, exit_thresh))

    # VAD state
    end_silence_frames = max(1, end_silence_ms // frame_ms)
    frame_samples = max(1, int(used_sr * frame_ms / 1000))
    in_speech = False
    utter_frames = []
    silence_run = 0
    utter_start_ts = None
    debounce_until = 0.0

    # Preroll & adaptive noise
    preroll_samples_target = max(0, int(used_sr * (preroll_ms / 1000.0)))
    prebuf = collections.deque(maxlen=max(1, preroll_samples_target // frame_samples + 1))
    noise_ema = noise_r
    ema_alpha = 0.05

    _flush_print("[Live] Listening… (Ctrl+C to stop)")
    while _running:
        frame = _gather_samples(q, needed=frame_samples, timeout_s=0.8)
        if frame.size == 0:
            continue

        frame_r = rms(frame)
        prebuf.append(frame)

        if not in_speech:
            noise_ema = (1.0 - ema_alpha) * noise_ema + ema_alpha * frame_r
            enter_thresh = max(DEFAULT_RMS_SILENCE_FLOOR, noise_ema * float(enter_mult))
            exit_thresh  = max(DEFAULT_RMS_SILENCE_FLOOR, noise_ema * float(exit_mult))

        now = time.time()
        if now < debounce_until:
            # cooldown after we just finished an utterance
            continue

        if frame_r > enter_thresh:
            if not in_speech:
                in_speech = True
                utter_frames = []
                utter_start_ts = now
                if len(prebuf):
                    utter_frames.extend(list(prebuf))
            utter_frames.append(frame.copy())
            silence_run = 0
            if (now - (utter_start_ts or now)) > max_utterance_sec:
                in_speech = False
        else:
            if in_speech:
                silence_run += 1
                utter_frames.append(frame.copy())
                if silence_run >= end_silence_frames:
                    in_speech = False

        # Completed utterance → process
        if (not in_speech) and utter_frames:
            total_samples = int(np.sum([len(f) for f in utter_frames]))
            if total_samples < int(used_sr * (min_utter_ms / 1000.0)):
                # Too short; drop + small debounce
                utter_frames = []
                silence_run = 0
                utter_start_ts = None
                debounce_until = time.time() + (debounce_ms / 1000.0)
                continue

            ts = time.strftime("%Y-%m-%d_%H-%M-%S")
            audio_f32 = np.concatenate(utter_frames).astype(np.float32)
            audio_i16 = np.clip(audio_f32 * 32767.0, -32768.0, 32767.0).astype(np.int16)
            wav_path = SESSION_DIR / ("utter_%s.wav" % ts)
            write_wav(wav_path, audio_i16, used_sr)
            _flush_print("[Live] Utterance -> %s" % wav_path)

            # Transcribe
            _flush_print("[Live] Transcribing…")
            text = (transcribe_file(str(wav_path)) or "").strip()
            _flush_print("[Live] You said: %s" % (text or "[empty]"))

            # Label speaker (enroll if they told us their name)
            label_from_text = _extract_name_from_text(text)
            if label_from_text:
                maybe = spk_id.best_label_for_name(label_from_text, min_ratio=0.78)
                if maybe:
                    canonical = min(maybe, label_from_text, key=len).title()
                    if canonical != maybe:
                        spk_id.rename(maybe, canonical)
                    label = canonical
                else:
                    spk_id.enroll(label_from_text, wav_path)
                    label = label_from_text.title()
                sim = 1.0
            else:
                label, sim = spk_id.identify(wav_path)

            # --- Recency-aware override to prevent "Guest-N" explosion ---
            overrode = False
            now_ts = time.time()
            if not label_from_text:
                # Only consider override if we (a) created a Guest-* or (b) ambiguity likely
                scores = spk_id.score(wav_path) if hasattr(spk_id, "score") else []
                if scores:
                    best_name, best_sim = scores[0]
                    second_sim = scores[1][1] if len(scores) > 1 else -1.0

                    made_guest = label.startswith("Guest-")
                    ambiguous = (best_sim - second_sim) < 0.08  # same as margin
                    recent_bias_ok = (best_sim >= 0.80) and (
                        best_name == _last_label or (now_ts - _last_seen.get(best_name, 0)) < 8.0
                    )

                    if made_guest and ambiguous and recent_bias_ok:
                        # Reuse the recent speaker and strengthen its centroid
                        if hasattr(spk_id, "add_sample_to"):
                            spk_id.add_sample_to(best_name, wav_path)
                        label = best_name
                        overrode = True

            _flush_print(f"[ID] Labeled: {label} (mode={spk_id.mode}, sim={sim:.2f}{' OVERRIDDEN' if overrode else ''})")

            # Update recency tables
            _last_seen[label] = now_ts
            _last_label = label

            # Compose speaker-aware prompt (+ per-speaker memory)
            participants = spk_id.labels()
            base_prompt = cfg.get("llm", {}).get(
                "base_system_prompt",
                "You are a concise, helpful multi-party assistant. Keep replies to 1–3 sentences. "
                "Do NOT guess the number or identity of speakers unless provided. "
                "Keep each thread separate; do not mix details across speakers. "
                
            )
            core = build_speaker_aware_system_prompt(
                base_prompt=base_prompt,
                participants=participants,
                last_speaker=label,
            )
            history = memory.format_for_prompt(label)
            system_prompt = core + (f"\nRecent context for {label}:\n{history}" if history and history != "None" else "")

            # LLM
            _flush_print("[Live] Thinking…")
            model_name = cfg.get("llm", {}).get("model", "gpt-3.5-turbo")
            reply = (generate_response(text, system_prompt=system_prompt, model_name=model_name) or "").strip()

            addressed = f"{label}, {reply}" if reply and not reply.lower().startswith(label.lower()) else reply
            _flush_print("[Live] Agent → %s: %s" % (label, addressed or "[empty]"))

            # Update memory AFTER generating the reply
            memory.add(label, user_text=text, agent_text=addressed or "")

            # Speak (pause mic), then resume
            stream.stop()
            try:
                voice_id = cfg.get("tts", {}).get("voice_id")
                model_id = cfg.get("tts", {}).get("model_id")
                speak_text(
                    addressed,
                    save_to=SESSION_DIR / ("utter_%s_reply.mp3" % ts),
                    play=bool(play_audio),
                    voice_id=voice_id,
                    model_id=model_id,
                )
            except Exception as e:
                _flush_print("[Live] TTS error (continuing): %s" % e)
            finally:
                try:
                    _drain_queue(q)
                    stream.start()
                except Exception as e:
                    _flush_print("[Live] Could not restart stream: %s" % e)
                    break

            # Reset + debounce
            utter_frames = []
            silence_run = 0
            utter_start_ts = None
            debounce_until = time.time() + (DEFAULT_DEBOUNCE_MS / 1000.0)

    try:
        stream.stop()
        stream.close()
    except Exception:
        pass
    _flush_print("\n[Live] Stopped.")

def parse_args():
    p = argparse.ArgumentParser(description="Realtime multi-speaker assistant (callback VAD + safe speaker labels)")
    p.add_argument("--device", type=int, default=None, help="Input device index (see python -m sounddevice)")
    p.add_argument("--samplerate", type=int, default=48000, help="Preferred sample rate (Hz)")
    p.add_argument("--frame-ms", type=int, default=DEFAULT_FRAME_MS, help="Analysis frame size (ms)")
    p.add_argument("--end-silence-ms", type=int, default=DEFAULT_END_SILENCE_MS, help="Silence to end utterance (ms)")
    p.add_argument("--max-utterance-sec", type=int, default=DEFAULT_MAX_UTTERANCE_SEC, help="Max utterance length (sec)")
    p.add_argument("--calibration-sec", type=float, default=DEFAULT_CALIBRATION_SEC, help="Calibration duration (sec)")
    p.add_argument("--enter-mult", type=float, default=DEFAULT_ENTER_MULT, help="RMS× to start speech")
    p.add_argument("--exit-mult", type=float, default=DEFAULT_EXIT_MULT, help="RMS× to end speech")
    p.add_argument("--no-play", action="store_true", help="Do not play audio aloud (still saves reply mp3)")
    p.add_argument("--preroll-ms", type=int, default=DEFAULT_PREROLL_MS, help="Audio to keep before speech start (ms)")
    p.add_argument("--min-utter-ms", type=int, default=DEFAULT_MIN_UTTER_MS, help="Discard utterances shorter than this (ms)")
    p.add_argument("--debounce-ms", type=int, default=DEFAULT_DEBOUNCE_MS, help="Cooldown after finishing (ms)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    realtime_loop(
        device=args.device,
        samplerate=args.samplerate,
        frame_ms=args.frame_ms,
        end_silence_ms=args.end_silence_ms,
        max_utterance_sec=args.max_utterance_sec,
        calibration_sec=args.calibration_sec,
        enter_mult=args.enter_mult,
        exit_mult=args.exit_mult,
        play_audio=not args.no_play,
        preroll_ms=args.preroll_ms,
        min_utter_ms=args.min_utter_ms,
        debounce_ms=args.debounce_ms,
    )

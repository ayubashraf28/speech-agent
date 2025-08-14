# mic_check.py
import argparse
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


def list_devices() -> None:
    try:
        devs = sd.query_devices()
        default_in, default_out = sd.default.device
        print(f"[MicCheck] Found {len(devs)} devices.")
        print(f"[MicCheck] Default input/output: {default_in} / {default_out}")
        for idx, d in enumerate(devs):
            io = []
            if d["max_input_channels"] > 0:
                io.append("IN")
            if d["max_output_channels"] > 0:
                io.append("OUT")
            print(f"  [{idx:02d}] {'/'.join(io):3s}  {d['name']}")
    except Exception as e:
        print(f"[MicCheck] Could not query devices: {e}")


def record(seconds: int, samplerate: int, channels: int, device: int | None) -> np.ndarray:
    try:
        if device is not None:
            sd.default.device = (device, sd.default.device[1])
        print(f"[MicCheck] Recording {seconds}s at {samplerate} Hz (ch={channels})… Speak now.")
        audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=channels, dtype="float32")
        sd.wait()
        print("[MicCheck] Recording complete.")
        return np.squeeze(audio)
    except Exception as e:
        print(f"[MicCheck] Record error: {e}")
        sys.exit(2)


def maybe_playback(audio: np.ndarray, samplerate: int, device: int | None) -> None:
    try:
        if device is not None:
            # keep input device as-is; set output device index if provided via --playback-device
            pass
        sd.play(audio, samplerate=samplerate)
        sd.wait()
        print("[MicCheck] Playback finished.")
    except Exception as e:
        print(f"[MicCheck] Playback error (skipping): {e}")


def write_wav(path: Path, audio: np.ndarray, samplerate: int) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # ensure mono vector for 1ch
        if audio.ndim > 1 and audio.shape[1] == 1:
            audio = np.squeeze(audio, axis=1)
        sf.write(str(path), audio, samplerate, subtype="PCM_16")
        print(f"[MicCheck] Saved -> {path.resolve()}")
        print(f"[MicCheck] Bytes: {path.stat().st_size}")
    except Exception as e:
        print(f"[MicCheck] Save error: {e}")
        sys.exit(3)


def parse_args():
    p = argparse.ArgumentParser(description="Microphone sanity check")
    p.add_argument("--list", action="store_true", help="List audio devices and exit")
    p.add_argument("--device", type=int, default=None, help="Input device index (see --list)")
    p.add_argument("--seconds", type=int, default=5, help="Record duration in seconds")
    p.add_argument("--samplerate", type=int, default=16000, help="Sample rate (Hz)")
    p.add_argument("--channels", type=int, default=1, help="Number of input channels")
    p.add_argument("--playback", action="store_true", help="Play back the recorded audio")
    p.add_argument("--out", type=Path, default=Path("data/live/check.wav"), help="Output WAV path")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        list_devices()
        return

    # Print a brief device summary up front (non-fatal if it fails)
    try:
        default = sd.default.device
        print(f"[MicCheck] Default devices (in/out): {default}")
    except Exception as e:
        print(f"[MicCheck] Could not read default devices: {e}")

    audio = record(args.seconds, args.samplerate, args.channels, args.device)
    write_wav(args.out, audio, args.samplerate)

    if args.playback:
        maybe_playback(audio, args.samplerate, None)


if __name__ == "__main__":
    main()

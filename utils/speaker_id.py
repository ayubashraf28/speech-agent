from __future__ import annotations
import difflib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

# Optional backends
_BACKEND = "none"
try:
    from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore
    _BACKEND = "resemblyzer"
except Exception:
    try:
        import torchaudio  # type: ignore
        import torch  # noqa: F401
        _BACKEND = "torchaudio"
    except Exception:
        _BACKEND = "none"


@dataclass
class _SpeakerEntry:
    centroid: np.ndarray          # L2-normalized running centroid
    count: int                    # number of utterances contributing
    last_update_ts: float         # time.time() at last update
    history: List[np.ndarray]     # small ring buffer for debugging


class SpeakerID:
    """
    Speaker identification with stabilization & persistence.

    - Embedding via Resemblyzer (preferred) or MFCC fallback.
    - Gating: min duration & min RMS.
    - EMA centroid update for each speaker.
    - Temporal smoothing: prefer last label if within margin of threshold.
    - New-speaker hysteresis before minting Guest-N.
    - Persistent NPZ at db_path.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.82,
        margin: float = 0.04,
        min_duration_sec: float = 1.0,
        min_rms: float = 8e-4,
        ema_alpha: float = 0.25,
        hysteresis_unknown: int = 2,
        db_path: Optional[Path] = None,
        history_keep: int = 6,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.threshold = float(threshold)
        self.margin = float(margin)
        self.min_duration_sec = float(min_duration_sec)
        self.min_rms = float(min_rms)
        self.ema_alpha = float(ema_alpha)
        self.hysteresis_unknown = int(hysteresis_unknown)
        self.history_keep = int(history_keep)

        self.mode = _BACKEND
        self._enc = None
        if self.mode == "resemblyzer":
            self._enc = VoiceEncoder()  # auto GPU/CPU

        # persistence
        self._store_dir = Path("data/speakers")
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = Path(db_path) if db_path else (self._store_dir / "default.npz")

        # runtime state
        self._speakers: Dict[str, _SpeakerEntry] = {}
        self._next_guest_idx: int = 1
        self._last_label: Optional[str] = None
        self._unknown_hits: int = 0

        self._load()  # best-effort

    # ---------- Public API ----------

    def labels(self) -> List[str]:
        return sorted(self._speakers.keys())

    def save(self, path: Optional[Path] = None) -> None:
        """Persist to NPZ. If path is given, update self._db_path and save there."""
        if path is not None:
            self._db_path = Path(path)
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._save()

    def load(self, path: Optional[Path] = None) -> None:
        """Load from NPZ. If path is given, update self._db_path and load from there."""
        if path is not None:
            self._db_path = Path(path)
        self._load()

    def reset(self) -> None:
        """Clear all speakers and persist."""
        self._speakers.clear()
        self._next_guest_idx = 1
        self._last_label = None
        self._unknown_hits = 0
        self._save()

    def best_label_for_name(self, name: str, min_ratio: float = 0.78) -> Optional[str]:
        """Fuzzy match for claimed names, case-insensitive."""
        if not name:
            return None
        name_t = name.strip().lower()
        if not name_t:
            return None
        best, best_r = None, 0.0
        for lab in self._speakers.keys():
            r = difflib.SequenceMatcher(None, lab.lower(), name_t).ratio()
            if r > best_r:
                best, best_r = lab, r
        return best if best_r >= float(min_ratio) else None

    def rename(self, old: str, new: str) -> None:
        if not old or not new or old == new or old not in self._speakers:
            return
        if new in self._speakers:
            a = self._speakers[old]
            b = self._speakers[new]
            merged = self._merge_entries(a, b)
            self._speakers[new] = merged
            del self._speakers[old]
        else:
            self._speakers[new] = self._speakers.pop(old)
        if self._last_label == old:
            self._last_label = new
        self._save()

    def enroll(self, label: str, wav_path: Path | str) -> bool:
        """Add/update a speaker with a gated embedding. Returns True if updated."""
        emb = self._embed_file(wav_path)
        if emb is None:
            return False
        label = label.strip().title()
        if not label:
            return False
        if label in self._speakers:
            self._update_centroid(label, emb)
        else:
            self._speakers[label] = _SpeakerEntry(
                centroid=self._l2(emb),
                count=1,
                last_update_ts=self._now(),
                history=[emb],
            )
        self._last_label = label
        self._unknown_hits = 0
        self._save()
        return True

    def identify(self, wav_path: Path | str) -> Tuple[str, float]:
        """
        Return (label, similarity).

        - If DB empty -> mint Guest-1 immediately.
        - Else compute embedding and match to best centroid.
        - Accept if sim >= threshold.
        - Else, if last_label is best and sim >= (threshold - margin) -> reuse last_label.
        - Else increment unknown hits; if >= hysteresis, mint Guest-N and enroll; otherwise return best label with low sim.
        """
        if not self._speakers:
            label = self._mint_guest_label()
            self.enroll(label, wav_path)
            return label, 1.0

        emb = self._embed_file(wav_path)
        if emb is None:
            if self._last_label is not None:
                return self._last_label, 0.0
            label = self._mint_guest_label()
            self.enroll(label, wav_path)
            return label, 0.0

        best_label, best_sim = self._best_match(emb)

        if best_sim >= self.threshold:
            self._last_label = best_label
            self._unknown_hits = 0
            self._update_centroid(best_label, emb)
            return best_label, best_sim

        if self._last_label == best_label and best_sim >= (self.threshold - self.margin):
            self._unknown_hits = 0
            self._update_centroid(best_label, emb)
            return best_label, best_sim

        self._unknown_hits += 1
        if self._unknown_hits >= self.hysteresis_unknown:
            label = self._mint_guest_label()
            self.enroll(label, wav_path)
            return label, best_sim

        return best_label, best_sim

    # ---------- Internals ----------

    def _merge_entries(self, a: _SpeakerEntry, b: _SpeakerEntry) -> _SpeakerEntry:
        ca, cb = a.centroid, b.centroid
        wa, wb = max(1, a.count), max(1, b.count)
        merged = self._l2((ca * wa + cb * wb) / float(wa + wb))
        hist = (a.history + b.history)[-self.history_keep:]
        return _SpeakerEntry(centroid=merged, count=wa + wb, last_update_ts=max(a.last_update_ts, b.last_update_ts), history=hist)

    def _now(self) -> float:
        import time as _t
        return _t.time()

    def _l2(self, x: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(x) + 1e-9)
        return (x / n).astype(np.float32, copy=False)

    def _cos(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _best_match(self, emb: np.ndarray) -> Tuple[str, float]:
        emb = self._l2(emb)
        best_l, best_s = None, -1.0
        for lab, entry in self._speakers.items():
            s = self._cos(entry.centroid, emb)
            if s > best_s:
                best_l, best_s = lab, s
        return best_l or "Guest-?", float(best_s)

    def _update_centroid(self, label: str, emb: np.ndarray) -> None:
        if label not in self._speakers:
            self._speakers[label] = _SpeakerEntry(
                centroid=self._l2(emb), count=1, last_update_ts=self._now(), history=[emb]
            )
        else:
            e = self._speakers[label]
            ema = float(self.ema_alpha)
            new_c = self._l2((1.0 - ema) * e.centroid + ema * self._l2(emb))
            e.centroid = new_c
            e.count += 1
            e.last_update_ts = self._now()
            e.history.append(emb)
            if len(e.history) > self.history_keep:
                e.history.pop(0)
        self._save()

    def _mint_guest_label(self) -> str:
        while True:
            label = f"Guest-{self._next_guest_idx}"
            self._next_guest_idx += 1
            if label not in self._speakers:
                return label

    # ---------- Embedding backends ----------

    def _embed_file(self, wav_path: Path | str) -> Optional[np.ndarray]:
        try:
            audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        except Exception:
            return None

        if audio.ndim == 2:
            audio = audio[:, 0]
        dur = len(audio) / float(sr if sr else self.sample_rate)
        if dur < self.min_duration_sec:
            return None

        rms = math.sqrt(float(np.mean(np.square(audio)))) if audio.size else 0.0
        if rms < self.min_rms:
            return None

        if self.mode == "resemblyzer":
            try:
                wav16 = preprocess_wav(Path(wav_path))  # 16k mono float32
                emb = self._enc.embed_utterance(wav16)  # type: ignore
                return self._l2(np.asarray(emb, dtype=np.float32))
            except Exception:
                pass

        if self.mode == "torchaudio":
            try:
                import torchaudio  # type: ignore
                wav, srr = torchaudio.load(str(wav_path))
                if wav.size(0) > 1:
                    wav = wav[:1, :]
                if int(srr) != self.sample_rate:
                    wav = torchaudio.functional.resample(wav, int(srr), self.sample_rate)
                mfcc = torchaudio.transforms.MFCC(sample_rate=self.sample_rate, n_mfcc=40)(wav)
                mfcc = mfcc.squeeze(0).numpy()
                emb = mfcc.mean(axis=1).astype(np.float32)
                return self._l2(emb)
            except Exception:
                pass

        return None

    # ---------- Persistence ----------

    def _save(self) -> None:
        try:
            labels = list(self._speakers.keys())
            if labels:
                cents = np.stack([self._speakers[l].centroid for l in labels], axis=0)
                counts = np.asarray([self._speakers[l].count for l in labels], dtype=np.int32)
            else:
                cents = np.zeros((0, 256), dtype=np.float32)
                counts = np.zeros((0,), dtype=np.int32)
            np.savez_compressed(
                str(self._db_path),
                labels=np.array(labels, dtype=object),
                centroids=cents,
                counts=counts,
                next_guest=int(self._next_guest_idx),
            )
        except Exception:
            # never crash on save
            pass

    def _load(self) -> None:
        if not self._db_path.exists():
            return
        try:
            data = np.load(str(self._db_path), allow_pickle=True)
            labels = [str(x) for x in data.get("labels", [])]
            cents = np.asarray(data.get("centroids", np.zeros((0, 256), dtype=np.float32)), dtype=np.float32)
            counts = np.asarray(data.get("counts", np.zeros((len(labels),), dtype=np.int32)), dtype=np.int32)
            self._speakers.clear()
            for i, lab in enumerate(labels):
                c = cents[i] if i < len(cents) else np.zeros((256,), dtype=np.float32)
                n = int(counts[i]) if i < len(counts) else 1
                self._speakers[lab] = _SpeakerEntry(
                    centroid=self._l2(c),
                    count=max(1, n),
                    last_update_ts=self._now(),
                    history=[],
                )
            ng = int(data.get("next_guest", 1))
            self._next_guest_idx = max(1, ng)
        except Exception:
            self._speakers.clear()
            self._next_guest_idx = 1

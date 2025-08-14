# utils/speaker_id.py
from __future__ import annotations
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    _HAVE_RESEMBLYZER = True
except Exception:
    _HAVE_RESEMBLYZER = False

# Small, dependency-free fuzzy match
from difflib import SequenceMatcher

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

@dataclass
class _LabelData:
    # store a rolling buffer of embeddings to keep centroid robust
    embs: List[np.ndarray] = field(default_factory=list)
    max_keep: int = 12

    def add(self, e: np.ndarray):
        self.embs.append(e.astype(np.float32, copy=False))
        if len(self.embs) > self.max_keep:
            # keep the most recent self.max_keep
            self.embs = self.embs[-self.max_keep:]

    def centroid(self) -> np.ndarray:
        if not self.embs:
            return np.zeros((256,), dtype=np.float32)
        m = np.mean(np.stack(self.embs, axis=0), axis=0)
        return m.astype(np.float32, copy=False)

class SpeakerID:
    """
    Deterministic, minimal resemblyzer wrapper:
      - identify(): returns (best_label, best_sim) if >= threshold, else (None, best_sim)
      - enroll(name): creates/updates a label with this utterance
      - add_sample_to(name): add sample to existing label
      - auto_enroll_guest(): creates "Guest-N" when voice is unknown
      - rename(old, new), labels(), best_label_for_name()

    No hidden 'auto-guest per turn' behavior; threshold governs reuse.
    """
    def __init__(
        self,
        threshold: float = 0.60,
        margin: float = 0.08,        # for future hysteresis; not heavily used here
        sample_rate: int = 16000,
        device: Optional[str] = None,
        max_keep_per_label: int = 12,
    ) -> None:
        if not _HAVE_RESEMBLYZER:
            raise RuntimeError("Resemblyzer not available")

        # VoiceEncoder will pick CUDA if available
        self.encoder = VoiceEncoder(device=device)
        self.sample_rate = sample_rate
        self.threshold = float(threshold)
        self.margin = float(margin)

        self._labels: Dict[str, _LabelData] = {}
        self._guest_counter = 0
        self._max_keep = int(max_keep_per_label)

    # ----- public info -----
    @property
    def mode(self) -> str:
        return "resemblyzer"

    def labels(self) -> List[str]:
        return sorted(self._labels.keys())

    # ----- I/O helpers -----
    def _embed_from_file(self, wav_path: Union[str, Path]) -> np.ndarray:
        wav = preprocess_wav(Path(wav_path))
        emb = self.encoder.embed_utterance(wav)  # (256,)
        return emb.astype(np.float32, copy=False)

    # ----- enrollment / updates -----
    def enroll(self, name: str, wav_path: Union[str, Path]) -> None:
        name = name.strip().title()
        emb = self._embed_from_file(wav_path)
        lb = self._labels.get(name)
        if lb is None:
            lb = _LabelData(max_keep=self._max_keep)
            self._labels[name] = lb
        lb.add(emb)

    def add_sample_to(self, name: str, wav_path_or_emb: Union[str, Path, np.ndarray]) -> None:
        name = name.strip().title()
        if isinstance(wav_path_or_emb, (str, Path)):
            emb = self._embed_from_file(wav_path_or_emb)
        else:
            emb = np.asarray(wav_path_or_emb, dtype=np.float32)
        lb = self._labels.get(name)
        if lb is None:
            lb = _LabelData(max_keep=self._max_keep)
            self._labels[name] = lb
        lb.add(emb)

    def auto_enroll_guest(self, wav_path: Union[str, Path]) -> str:
        self._guest_counter += 1
        name = f"Guest-{self._guest_counter}"
        self.enroll(name, wav_path)
        return name

    def rename(self, old: str, new: str) -> None:
        old = old.strip().title()
        new = new.strip().title()
        if old == new:
            return
        if old in self._labels:
            if new in self._labels:
                # merge old into new
                for e in self._labels[old].embs:
                    self._labels[new].add(e)
                del self._labels[old]
            else:
                self._labels[new] = self._labels.pop(old)

    # ----- scoring / identification -----
    def score(self, wav_path_or_emb: Union[str, Path, np.ndarray]) -> List[Tuple[str, float]]:
        if isinstance(wav_path_or_emb, (str, Path)):
            q = self._embed_from_file(wav_path_or_emb)
        else:
            q = np.asarray(wav_path_or_emb, dtype=np.float32)

        scores: List[Tuple[str, float]] = []
        for name, lb in self._labels.items():
            c = lb.centroid()
            sim = _cosine_sim(q, c)
            scores.append((name, float(sim)))
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores

    def identify(self, wav_path: Union[str, Path]) -> Tuple[Optional[str], float]:
        scores = self.score(wav_path)
        if not scores:
            return None, 0.0
        top_name, top_sim = scores[0]
        if top_sim >= self.threshold:
            return top_name, top_sim
        return None, top_sim

    # ----- name helpers -----
    def best_label_for_name(self, claimed: str, min_ratio: float = 0.88) -> Optional[str]:
        claimed_t = claimed.strip().title()
        best_name = None
        best_ratio = 0.0
        for name in self._labels.keys():
            r = SequenceMatcher(a=claimed_t, b=name).ratio()
            if r > best_ratio:
                best_ratio = r
                best_name = name
        if best_name and best_ratio >= float(min_ratio):
            return best_name
        return None

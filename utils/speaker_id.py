# utils/speaker_id.py
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, Tuple, List, Optional
from difflib import SequenceMatcher

# NOTE: Do NOT import torch/torchaudio at module import time (Windows safety).
# We'll lazy-import them inside the class, and fall back if unavailable.


class SpeakerID:
    """
    Lightweight speaker labelling with a safe fallback.

    Modes:
      - "torchaudio": MFCC + cosine, with threshold + top-2 margin.
      - "fallback":  names enroll; otherwise assign Guest-N (no embeddings).

    Public API:
      - enroll(name, wav_path)
      - identify(wav_path) -> (label, similarity)
      - best_label_for_name(name)
      - rename(old_label, new_label)
      - labels() -> List[str]
      - mode -> "torchaudio" | "fallback"
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        enroll_max: int = 5,
        threshold: float = 0.68,
        margin: float = 0.08,
    ):
        self.sr = int(sample_rate)
        self.n_mfcc = int(n_mfcc)
        self.threshold = float(threshold)
        self.margin = float(margin)

        # state shared by both modes
        self._guest_counter = 1
        self._labels_order: List[str] = []  # preserve insertion order for labels()

        # torchaudio objects (lazy)
        self._ok = False
        self._torch = None
        self._torchaudio = None
        self._mfcc = None
        self._resampler = None

        # embeddings store (torchaudio mode only)
        self._buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=enroll_max))
        self._centroids: Dict[str, "object"] = {}

        # Try to init torchaudio path
        self._try_init_torchaudio()

    # -------------------- mode & helpers --------------------

    @property
    def mode(self) -> str:
        return "torchaudio" if self._ok else "fallback"

    def _try_init_torchaudio(self):
        try:
            import torch  # lazy
            import torchaudio  # lazy
            self._torch = torch
            self._torchaudio = torchaudio
            self._mfcc = torchaudio.transforms.MFCC(
                sample_rate=self.sr,
                n_mfcc=self.n_mfcc,
                melkwargs={"n_mels": 64},
            )
            self._ok = True
        except Exception:
            # Stay in fallback mode
            self._ok = False
            self._torch = None
            self._torchaudio = None
            self._mfcc = None
            self._resampler = None

    def _ensure_label_order(self, label: str):
        if label not in self._labels_order:
            self._labels_order.append(label)

    # -------------------- public API --------------------

    def score(self, wav_path: Path) -> List[Tuple[str, float]]:
        """ Return ranked [(label, sim), ...] of known speakers. Fallback mode -> empty list."""
        if not self._ok or not self._centroids:
            return []
        emb = self._embed_file(wav_path)
        torch = self._torch
        sims = [(name, float(torch.dot(emb, c))) for name, c in self._centroids.items()]
        sims.sort(key=lambda kv: kv[1], reverse=True)
        return sims

    def add_sample_to(self, label: str, wav_path: Path) -> None:
        """
        Force-attach this audio sample to an existing label and update its centroid.
        No-op in fallback mode.
        """
        if not self._ok:
            return
        label = (label or "").strip().title()
        if not label or label not in self._buffers:
            return
        emb = self._embed_file(wav_path)
        self._buffers[label].append(emb)
        self._centroids[label] = self._mean_centroid(self._buffers[label])


    def enroll(self, name: str, wav_path: Path) -> None:
        name = (name or "").strip().title()
        if not name:
            return
        self._ensure_label_order(name)
        if not self._ok:
            # fallback: just remember the label
            return
        emb = self._embed_file(wav_path)
        self._buffers[name].append(emb)
        self._centroids[name] = self._mean_centroid(self._buffers[name])

    def best_label_for_name(self, name: str, min_ratio: float = 0.78) -> str:
        """Find an existing label that looks like this name (for typo/mishearing correction)."""
        name = (name or "").strip().title()
        best, score = "", 0.0
        for lbl in self.labels():
            r = SequenceMatcher(None, lbl, name).ratio()
            if r > score:
                best, score = lbl, r
        return best if score >= float(min_ratio) else ""

    def rename(self, old_label: str, new_label: str) -> bool:
        old_label = (old_label or "").strip()
        new_label = (new_label or "").strip().title()
        if not old_label or not new_label:
            return False
        if old_label == new_label:
            return True

        # update ordering
        if old_label in self._labels_order:
            self._labels_order = [new_label if x == old_label else x for x in self._labels_order]
            if new_label not in self._labels_order:
                self._labels_order.append(new_label)

        if not self._ok:
            # fallback: nothing else to merge
            return True

        if old_label not in self._buffers:
            return False
        buf_old = self._buffers.pop(old_label)
        if new_label not in self._buffers:
            self._buffers[new_label] = deque(maxlen=buf_old.maxlen)
        for emb in list(buf_old):
            self._buffers[new_label].append(emb)
        self._centroids[new_label] = self._mean_centroid(self._buffers[new_label])
        self._centroids.pop(old_label, None)
        return True

    def identify(self, wav_path: Path) -> Tuple[str, float]:
        """Return (label, similarity). In fallback mode, always create a new Guest-N."""
        if not self._ok:
            label = self._new_guest()
            return label, 0.0

        emb = self._embed_file(wav_path)
        if not self._centroids:
            label = self._new_guest()
            # seed centroid for guest with first emb
            self._buffers[label].append(emb)
            self._centroids[label] = self._mean_centroid(self._buffers[label])
            return label, 0.0

        torch = self._torch
        sims = {name: float(torch.dot(emb, c)) for name, c in self._centroids.items()}
        ordered = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        best_name, best_sim = ordered[0]
        second_sim = ordered[1][1] if len(ordered) > 1 else -1.0

        if (best_sim >= self.threshold) and ((best_sim - second_sim) >= self.margin):
            # accept + online update
            self._buffers[best_name].append(emb)
            self._centroids[best_name] = self._mean_centroid(self._buffers[best_name])
            return best_name, best_sim

        # ambiguous -> new guest (seed centroid)
        label = self._new_guest()
        self._buffers[label].append(emb)
        self._centroids[label] = self._mean_centroid(self._buffers[label])
        return label, best_sim

    def labels(self) -> List[str]:
        # union of labels from order list + centroid keys (robust)
        seen = set()
        result = []
        for x in self._labels_order:
            if x not in seen:
                result.append(x); seen.add(x)
        if self._ok:
            for x in self._centroids.keys():
                if x not in seen:
                    result.append(x); seen.add(x)
        return result

    # -------------------- internals (torchaudio mode) --------------------

    def _new_guest(self) -> str:
        label = f"Guest-{self._guest_counter}"
        self._guest_counter += 1
        self._ensure_label_order(label)
        return label

    def _load_wav(self, path: Path):
        torchaudio = self._torchaudio
        wav, sr = torchaudio.load(str(path))
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sr:
            if (self._resampler is None) or (getattr(self._resampler, "orig_freq", None) != sr):
                self._resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)
            wav = self._resampler(wav)
        return wav  # (1, T)

    def _energy_trim(self, wav, pad_ms: float = 40.0):
        torch = self._torch
        x = wav.squeeze(0)
        env = x.abs()
        thr = max(0.005, float(env.median().item()) * 3.0)
        idx = (env > thr).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return wav
        sr = self.sr
        pad = int(sr * (pad_ms / 1000.0))
        start = max(0, int(idx[0].item()) - pad)
        end = min(x.numel(), int(idx[-1].item()) + pad)
        return x[start:end].unsqueeze(0)

    def _center_crop(self, wav, max_sec: float = 1.2):
        T = wav.size(1)
        max_len = int(self.sr * max_sec)
        if T <= max_len:
            return wav
        start = (T - max_len) // 2
        end = start + max_len
        return wav[:, start:end]

    def _embed_tensor(self, wav):
        torch = self._torch
        mfcc = self._mfcc
        with torch.no_grad():
            wav = self._energy_trim(wav)
            wav = self._center_crop(wav, max_sec=1.2)
            mf = mfcc(wav)                      # (1, n_mfcc, T)
            feat = mf.squeeze(0).transpose(0, 1)  # (T, n_mfcc)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            emb = feat.mean(dim=0)              # (n_mfcc,)
            emb = torch.nn.functional.normalize(emb, p=2, dim=0)
            return emb

    def _embed_file(self, path: Path):
        wav = self._load_wav(Path(path))
        return self._embed_tensor(wav)

    def _mean_centroid(self, buf: deque):
        torch = self._torch
        stack = torch.stack(list(buf), dim=0)
        return torch.nn.functional.normalize(stack.mean(dim=0), p=2, dim=0)

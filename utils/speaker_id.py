# utils/speaker_id.py
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, Tuple, List

import torch
import torchaudio

class SpeakerID:
    """
    Super-light speaker labelling:
    - MFCC embedding averaged over time, L2-normalized
    - Enrollment keeps a small deque of embeddings per name; centroid is used
    - Identification is cosine similarity to known centroids
    - If below threshold, creates a new Guest-N label
    NOTE: This is heuristic, not a production speaker-verification model.
    """

    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 40, enroll_max: int = 5, threshold: float = 0.60):
        self.sr = sample_rate
        self.n_mfcc = n_mfcc
        self.threshold = float(threshold)
        self._mfcc = torchaudio.transforms.MFCC(sample_rate=self.sr, n_mfcc=n_mfcc, melkwargs={"n_mels": 64})
        self._resampler = None  # built on first need
        self._buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=enroll_max))  # name -> deque[tensor]
        self._centroids: Dict[str, torch.Tensor] = {}  # name -> tensor
        self._guest_counter = 1

    # -------- public API --------
    def enroll(self, name: str, wav_path: Path) -> None:
        name = name.strip()
        if not name:
            return
        emb = self._embed_file(wav_path)
        self._buffers[name].append(emb)
        self._centroids[name] = self._mean_centroid(self._buffers[name])

    def identify(self, wav_path: Path) -> Tuple[str, float]:
        """Return (label, similarity). Creates a new Guest-N if no match."""
        emb = self._embed_file(wav_path)
        if not self._centroids:
            label = f"Guest-{self._guest_counter}"
            self._guest_counter += 1
            self._buffers[label].append(emb)
            self._centroids[label] = self._mean_centroid(self._buffers[label])
            return label, 0.0

        sims = {name: float(torch.dot(emb, c)) for name, c in self._centroids.items()}
        best_name, best_sim = max(sims.items(), key=lambda kv: kv[1])
        if best_sim >= self.threshold:
            # Online update: refine centroid with this new sample
            self._buffers[best_name].append(emb)
            self._centroids[best_name] = self._mean_centroid(self._buffers[best_name])
            return best_name, best_sim

        # New guest
        label = f"Guest-{self._guest_counter}"
        self._guest_counter += 1
        self._buffers[label].append(emb)
        self._centroids[label] = self._mean_centroid(self._buffers[label])
        return label, best_sim

    def labels(self) -> List[str]:
        return list(self._centroids.keys())

    # -------- internals --------
    def _load_wav(self, path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(str(path))
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sr:
            if (self._resampler is None) or (getattr(self._resampler, "orig_freq", None) != sr):
                self._resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)
            wav = self._resampler(wav)
        return wav  # shape: (1, T)

    def _embed_tensor(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mfcc = self._mfcc(wav)            # (1, n_mfcc, T)
            feat = mfcc.squeeze(0).transpose(0, 1)  # (T, n_mfcc)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            emb = feat.mean(dim=0)            # (n_mfcc,)
            emb = torch.nn.functional.normalize(emb, p=2, dim=0)
            return emb

    def _embed_file(self, path: Path) -> torch.Tensor:
        wav = self._load_wav(Path(path))
        return self._embed_tensor(wav)

    @staticmethod
    def _mean_centroid(buf: deque) -> torch.Tensor:
        stack = torch.stack(list(buf), dim=0)
        return torch.nn.functional.normalize(stack.mean(dim=0), p=2, dim=0)

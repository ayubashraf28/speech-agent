# utils/persistent_memory.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

class PersistentSpeakerMemory:
    """
    File-backed per-speaker short-term memory.
    Stores the last N (user, agent) turns per *named* speaker.
    """
    def __init__(self, path: Path, max_turns: int = 3):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_turns = max_turns
        self._data: Dict[str, List[Dict[str, str]]] = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def save(self) -> None:
        try:
            self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def reset(self) -> None:
        self._data = {}
        try:
            if self.path.exists():
                self.path.unlink()
        except Exception:
            pass

    @staticmethod
    def _is_named(label: str) -> bool:
        if not label:
            return False
        L = label.strip().lower()
        return not (L.startswith("guest") or L == "unknown")

    def add(self, label: str, user_text: str, agent_text: str) -> None:
        if not self._is_named(label):
            return
        buf = self._data.setdefault(label, [])
        buf.append({"user": user_text or "", "agent": agent_text or ""})
        if len(buf) > self.max_turns:
            del buf[0 : len(buf) - self.max_turns]

    def format_for_prompt(self, label: str) -> str:
        if not self._is_named(label):
            return ""
        turns = self._data.get(label, [])
        if not turns:
            return ""
        lines = []
        for t in turns[-self.max_turns:]:
            u = (t.get("user") or "").replace("\n", " ").strip()
            a = (t.get("agent") or "").replace("\n", " ").strip()
            if u:
                lines.append(f"- {label} said: {u}")
            if a:
                lines.append(f"- Agent replied: {a}")
        return "\n".join(lines)

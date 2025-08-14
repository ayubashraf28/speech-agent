# utils/memory.py
from collections import deque
from typing import Deque, Dict, List, Tuple


class SpeakerMemory:
    """
    Keeps the last N user/agent turns per speaker.

    Example:
        mem = SpeakerMemory(max_turns=3)
        mem.add("spk1", "Hello", "Hi, how are you?")
        mem.add("spk1", "What's up?", "Just working on code.")
        print(mem.format_for_prompt("spk1"))
    """

    def __init__(self, max_turns: int = 3):
        self.max_turns = max_turns
        self._mem: Dict[str, Deque[Tuple[str, str]]] = {}

    def add(self, speaker: str, user_text: str, agent_text: str) -> None:
        """
        Add a new user/agent exchange for a speaker.
        Oldest entries are dropped once max_turns is exceeded.
        """
        q = self._mem.setdefault(speaker, deque(maxlen=self.max_turns))
        q.append((user_text.strip(), agent_text.strip()))

    def context_pairs(self, speaker: str) -> List[Tuple[str, str]]:
        """
        Retrieve stored (user_text, agent_text) pairs for a speaker.
        """
        return list(self._mem.get(speaker, []))

    def format_for_prompt(self, speaker: str) -> str:
        """
        Return a compact text block for LLM prompts, showing speaker history.
        """
        pairs = self.context_pairs(speaker)
        if not pairs:
            return "None"

        lines: List[str] = []
        for i, (u, a) in enumerate(pairs, 1):
            lines.append(f"- Turn {i} · USER: {u}")
            lines.append(f"          · AGENT: {a}")
        return "\n".join(lines)

    def reset(self, speaker: str) -> None:
        """
        Clear memory for a single speaker.
        """
        if speaker in self._mem:
            self._mem[speaker].clear()

    def reset_all(self) -> None:
        """
        Clear memory for all speakers.
        """
        self._mem.clear()

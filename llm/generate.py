# llm/generate.py
import os
from typing import Iterable, List, Optional

from dotenv import load_dotenv

# Do not import OpenAI client at module import if key might be missing.
# We'll import inside generate_response() after checking the key.
load_dotenv()

_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")


def build_speaker_aware_system_prompt(
    base_prompt: str,
    participants: Optional[Iterable[str]] = None,
    last_speaker: Optional[str] = None,
) -> str:
    """
    Augment the base system prompt with diarization context.
    """
    parts: List[str] = [base_prompt.strip()]
    if participants:
        parts.append("Participants detected: " + ", ".join(sorted(set(participants))))
    if last_speaker:
        parts.append(f"Last speaker: {last_speaker}. Address them naturally.")
    # Keep responses concise and spoken-friendly
    parts.append("Keep replies brief (1–3 sentences). Avoid lists unless asked.")
    return "\n".join(parts)


def generate_response(
    user_text: str,
    system_prompt: str = "You are a helpful speech agent.",
    model_name: str = _DEFAULT_MODEL,
    max_tokens: int = 120,
    temperature: float = 0.3,
) -> str:
    """
    Call the OpenAI chat completion with a concise, speech-friendly style.
    Gracefully degrades if OPENAI_API_KEY is not set or if the API call fails.
    """
    if not (user_text or "").strip():
        return "[No input detected]"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Degrade gracefully; the rest of the pipeline should still run.
        return "[LLM unavailable: set OPENAI_API_KEY in your environment]"

    # Import here so environments without the package / key don't crash on import.
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        return f"[LLM unavailable: OpenAI client not importable ({e})]"

    client = OpenAI(api_key=api_key)

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        # Don't raise—return a readable error string so the session completes.
        return f"[LLM error: {e}]"

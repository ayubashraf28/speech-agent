# llm/generate.py
from typing import Optional, List
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=api_key)

def generate_response(
    prompt: str,
    system_prompt: str = "You are a helpful speech agent.",
    model_name: Optional[str] = None,
) -> str:
    model_name = model_name or DEFAULT_MODEL
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM] Error generating response: {e}")
        return "[Error: Unable to generate response]"

def build_speaker_aware_system_prompt(
    base_prompt: str,
    participants: List[str],
    last_speaker: Optional[str],
) -> str:
    """
    Compose a system prompt that orients the model to the group context.
    Keeps replies concise and acknowledges the latest speaker.
    """
    parts = [base_prompt.strip()]
    if participants:
        parts.append(f"\nParticipants detected: {', '.join(sorted(set(participants)))}.")
    if last_speaker:
        parts.append(f"Most recent speaker: {last_speaker}.")
        parts.append("Address and acknowledge the most recent speaker first.")
    parts.append("Keep replies concise (1–3 short sentences).")
    return "\n".join(parts)

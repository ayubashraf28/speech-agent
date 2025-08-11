# llm/generate.py
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load API key and settings
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")  

if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

# Initialize client once
client = OpenAI(api_key=api_key)

def generate_response(prompt: str, system_prompt: str = "You are a helpful speech agent.") -> str:
    """
    Generate a response from the LLM given a user prompt.
    
    Args:
        prompt (str): The user input or transcribed speech.
        system_prompt (str): System instruction to guide the assistant's tone & behavior.
    
    Returns:
        str: Model's reply.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM] Error generating response: {e}")
        return "[Error: Unable to generate response]"

if __name__ == "__main__":
    # Test run
    reply = generate_response("Tell me a fun fact about AI.")
    print("GPT says:", reply)

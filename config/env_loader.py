# config/env_loader.py

from dotenv import load_dotenv
import os

def load_environment():
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    football_api_key = os.getenv("FOOTBALL_API_KEY")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in environment.")
    if not football_api_key:
        raise ValueError("FOOTBALL_API_KEY not found in environment.")

    return {
        "OPENAI_API_KEY": openai_key,
        "FOOTBALL_API_KEY": football_api_key
    }

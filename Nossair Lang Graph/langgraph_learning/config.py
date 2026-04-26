"""
Centralized configuration for LangGraph learning lessons.
Reads environment variables from .env file.
"""

import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

def get_ollama_model() -> str:
    """
    Get the Ollama model name from environment variable.
    Defaults to 'llama3.2' if not set.
    
    Returns:
        str: The model name (e.g., 'llama3.2', 'qwen3:0.6b')
    """
    return os.getenv("OLLAMA_MODEL", "llama3.2")


def get_temperature(default: float = 0.0) -> float:
    """
    Get temperature setting from environment variable.
    Defaults to provided value if not set.
    
    Args:
        default: Default temperature value
        
    Returns:
        float: Temperature value
    """
    temp_str = os.getenv("LLM_TEMPERATURE")
    if temp_str is not None:
        try:
            return float(temp_str)
        except ValueError:
            return default
    return default

from typing import Optional
import os
from .base import LLMProvider
from .gemini import GeminiProvider

def get_llm_provider(provider_name: str, **kwargs) -> LLMProvider:
    """
    Factory function to get an LLM provider instance.
    
    Args:
        provider_name (str): The name of the provider (e.g., "gemini").
        **kwargs: Configuration arguments for the provider.
        
    Returns:
        LLMProvider: An instance of the requested provider.
        
    Raises:
        ValueError: If the provider is unknown or configuration is missing.
    """
    if provider_name.lower() == "gemini":
        return GeminiProvider(**kwargs)
    elif provider_name.lower() == "openai":
        from .openai import OpenAIProvider
        return OpenAIProvider(**kwargs)
    
    raise ValueError(f"Unknown provider: {provider_name}")

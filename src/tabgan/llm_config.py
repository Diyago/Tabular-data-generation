# -*- coding: utf-8 -*-
"""
LLM API Configuration for external text generation via API endpoints.

Supports LM Studio, OpenAI-compatible APIs, and other local/remote LLM services.
"""

import os
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class LLMAPIConfig:
    """Configuration for LLM API-based text generation.
    
    Examples:
        # LM Studio local server
        config = LLMAPIConfig(
            base_url="http://localhost:1234",
            chat_url="http://localhost:1234/api/v1/chat",
            model="google/gemma-3-12b",
            timeout=90
        )
        
        # OpenAI API
        config = LLMAPIConfig(
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4",
            timeout=60
        )
        
        # Ollama local server
        config = LLMAPIConfig(
            base_url="http://localhost:11434",
            chat_url="http://localhost:11434/api/generate",
            model="llama3",
            timeout=120
        )
    """
    
    # API Endpoint configuration
    base_url: str = "http://localhost:1234"
    chat_url: Optional[str] = None
    
    # Model and authentication
    model: str = "google/gemma-3-12b"
    api_key: Optional[str] = None
    
    # Request settings
    timeout: int = 90
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    
    # Generation settings
    system_prompt: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """Set default chat_url if not provided."""
        if self.chat_url is None:
            # Default LM Studio chat endpoint
            self.chat_url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
    
    @classmethod
    def from_lm_studio(cls, 
                       base_url: str = "http://localhost:1234",
                       model: str = "google/gemma-3-12b",
                       timeout: int = 90,
                       **kwargs) -> "LLMAPIConfig":
        """Create config for LM Studio server."""
        return cls(
            base_url=base_url,
            model=model,
            timeout=timeout,
            **kwargs
        )
    
    @classmethod
    def from_openai(cls,
                      api_key: Optional[str] = None,
                      model: str = "gpt-4",
                      timeout: int = 60,
                      **kwargs) -> "LLMAPIConfig":
        """Create config for OpenAI API."""
        return cls(
            base_url="https://api.openai.com/v1",
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model=model,
            timeout=timeout,
            **kwargs
        )
    
    @classmethod
    def from_ollama(cls,
                     base_url: str = "http://localhost:11434",
                     model: str = "llama3",
                     timeout: int = 120,
                     **kwargs) -> "LLMAPIConfig":
        """Create config for Ollama server."""
        return cls(
            base_url=base_url,
            chat_url=f"{base_url.rstrip('/')}/api/generate",
            model=model,
            timeout=timeout,
            **kwargs
        )
    
    def get_headers(self) -> dict:
        """Get HTTP headers for API requests."""
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


# Legacy-style constants (for backward compatibility with config.py patterns)
LM_STUDIO_BASE_URL = "http://localhost:1234"
LM_STUDIO_CHAT_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_MODEL = "google/gemma-3-12b"
LM_STUDIO_TIMEOUT = 90

# OpenAI constants
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4"
OPENAI_TIMEOUT = 60

# Ollama constants  
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
OLLAMA_TIMEOUT = 120

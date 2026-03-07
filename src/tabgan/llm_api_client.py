# -*- coding: utf-8 -*-
"""
LLM API Client for external text generation via API endpoints.
"""

import logging
import json
from typing import Optional, Dict, Any, List
import requests

from tabgan.llm_config import LLMAPIConfig


class LLMAPIClient:
    """Client for generating text via external LLM APIs (LM Studio, OpenAI, Ollama, etc.).
    
    This client provides a unified interface for API-based text generation
    that can be used alongside or instead of local models.
    
    Example:
        from tabgan.llm_config import LLMAPIConfig
        from tabgan.llm_api_client import LLMAPIClient
        
        # LM Studio
        config = LLMAPIConfig.from_lm_studio(
            base_url="http://localhost:1234",
            model="google/gemma-3-12b"
        )
        client = LLMAPIClient(config)
        
        text = client.generate("Generate a name for a female engineer, Age: 30: ")
    """
    
    def __init__(self, config: Optional[LLMAPIConfig] = None):
        """
        Initialize the API client with configuration.
        
        Args:
            config: LLMAPIConfig instance. If None, uses default LM Studio config.
        """
        self.config = config or LLMAPIConfig()
        self.session = requests.Session()
        
    def generate(self, 
                 prompt: str,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 system_prompt: Optional[str] = None) -> str:
        """
        Generate text from a prompt using the configured API.
        
        Args:
            prompt: The text prompt to send to the LLM
            max_tokens: Maximum tokens to generate (overrides config)
            temperature: Sampling temperature (overrides config)
            system_prompt: Optional system prompt (overrides config)
            
        Returns:
            Generated text string
            
        Raises:
            requests.RequestException: If the API request fails
        """
        headers = self.config.get_headers()
        
        # Build request payload based on API type
        if "ollama" in self.config.chat_url or "11434" in self.config.chat_url:
            payload = self._build_ollama_payload(prompt, max_tokens, temperature, system_prompt)
        else:
            # Default to OpenAI-compatible format (LM Studio, OpenAI, etc.)
            payload = self._build_openai_payload(prompt, max_tokens, temperature, system_prompt)
        
        try:
            response = self.session.post(
                self.config.chat_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return self._extract_response_text(result)
            
        except requests.RequestException as e:
            logging.error(f"LLM API request failed: {e}")
            raise
        except (KeyError, json.JSONDecodeError) as e:
            logging.error(f"Failed to parse LLM API response: {e}")
            raise
            
    def _build_openai_payload(self, 
                              prompt: str,
                              max_tokens: Optional[int],
                              temperature: Optional[float],
                              system_prompt: Optional[str]) -> Dict[str, Any]:
        """Build OpenAI-compatible API request payload."""
        messages: List[Dict[str, str]] = []
        
        # Add system message if provided
        sys_prompt = system_prompt or self.config.system_prompt
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature or self.config.temperature,
            "top_p": self.config.top_p,
        }
    
    def _build_ollama_payload(self,
                              prompt: str,
                              max_tokens: Optional[int],
                              temperature: Optional[float],
                              system_prompt: Optional[str]) -> Dict[str, Any]:
        """Build Ollama API request payload."""
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
            }
        }
        
        # Add system prompt if provided
        sys_prompt = system_prompt or self.config.system_prompt
        if sys_prompt:
            payload["system"] = sys_prompt
            
        # Ollama uses num_predict for max tokens
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        elif self.config.max_tokens:
            payload["options"]["num_predict"] = self.config.max_tokens
            
        return payload
    
    def _extract_response_text(self, result: Dict[str, Any]) -> str:
        """Extract generated text from API response."""
        # OpenAI-compatible format
        if "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            if "message" in choice:
                return choice["message"].get("content", "").strip()
            elif "text" in choice:
                return choice["text"].strip()
        
        # Ollama format
        if "response" in result:
            return result["response"].strip()
        
        # Fallback: try to find any string content
        logging.warning(f"Unexpected API response format: {result}")
        return str(result)
    
    def generate_batch(self,
                       prompts: List[str],
                       max_tokens: Optional[int] = None,
                       temperature: Optional[float] = None) -> List[str]:
        """
        Generate text for multiple prompts sequentially.
        
        Args:
            prompts: List of prompts to generate from
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            
        Returns:
            List of generated text strings
        """
        results = []
        for i, prompt in enumerate(prompts):
            try:
                text = self.generate(prompt, max_tokens, temperature)
                results.append(text)
            except requests.RequestException as e:
                logging.error(f"Failed to generate for prompt {i}: {e}")
                results.append("")
        return results
    
    def check_connection(self) -> bool:
        """
        Check if the API endpoint is accessible.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to get models list or just check if server responds
            if "ollama" in self.config.base_url or "11434" in self.config.base_url:
                test_url = f"{self.config.base_url.rstrip('/')}/api/tags"
            else:
                # OpenAI-compatible: try /models endpoint
                test_url = f"{self.config.base_url.rstrip('/')}/v1/models"
            
            response = self.session.get(
                test_url, 
                headers=self.config.get_headers(),
                timeout=5
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close session."""
        self.session.close()
        return False

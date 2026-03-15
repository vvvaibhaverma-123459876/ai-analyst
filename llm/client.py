"""
llm/client.py
Provider-agnostic LLM client.
Supports: openai | anthropic
Add new providers by extending _call_<provider>().
"""

import os
from core.config import config
from core.exceptions import LLMError
from core.logger import get_logger

logger = get_logger(__name__)


class LLMClient:

    def __init__(self, provider: str = None, model: str = None, temperature: float = None):
        self.provider = (provider or config.LLM_PROVIDER).lower()
        self.model = model or config.LLM_MODEL
        self.temperature = temperature if temperature is not None else config.LLM_TEMPERATURE

    def complete(self, system: str, user: str) -> str:
        """
        Single-turn completion.
        Returns the assistant's reply as a plain string.
        """
        logger.info(f"LLM call | provider={self.provider} | model={self.model}")
        if self.provider == "openai":
            return self._call_openai(system, user)
        elif self.provider == "anthropic":
            return self._call_anthropic(system, user)
        else:
            raise LLMError(f"Unknown LLM provider: '{self.provider}'. "
                           f"Set LLM_PROVIDER=openai or anthropic in .env")

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------

    def _call_openai(self, system: str, user: str) -> str:
        api_key = config.OPENAI_API_KEY
        if not api_key:
            raise LLMError("OPENAI_API_KEY not set in .env")
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise LLMError(f"OpenAI call failed: {e}") from e

    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------

    def _call_anthropic(self, system: str, user: str) -> str:
        api_key = config.ANTHROPIC_API_KEY
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY not set in .env")
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return message.content[0].text.strip()
        except Exception as e:
            raise LLMError(f"Anthropic call failed: {e}") from e

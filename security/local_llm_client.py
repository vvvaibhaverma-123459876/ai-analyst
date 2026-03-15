"""
security/local_llm_client.py
Local LLM client via Ollama.
Used when internet_off_mode=True or local_llm_mode=True in policy.
Zero data leaves the machine.

Requires: pip install ollama
Or manual install: https://ollama.ai
"""

from __future__ import annotations
from core.logger import get_logger

logger = get_logger(__name__)


class LocalLLMClient:

    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url

    def complete(self, system: str, user: str) -> str:
        """
        Same interface as llm/client.py LLMClient.complete().
        Drop-in replacement when local mode is active.
        """
        try:
            import ollama
            response = ollama.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response["message"]["content"].strip()
        except ImportError:
            return self._http_fallback(system, user)
        except Exception as e:
            logger.error(f"Local LLM call failed: {e}")
            raise

    def _http_fallback(self, system: str, user: str) -> str:
        """Direct HTTP call to Ollama API if ollama package not installed."""
        import requests
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        resp = requests.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def is_available(self) -> bool:
        try:
            import requests
            resp = requests.get(f"{self._base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

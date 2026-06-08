"""
llm_client.py
-------------
Communicates with the LM Studio local server via its OpenAI-compatible API.
LM Studio exposes: http://localhost:1234/v1
"""

import logging
from typing import List, Dict, Generator

from openai import OpenAI

logger = logging.getLogger(__name__)

# LM Studio default endpoint — change in .env if needed
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"  # LM Studio ignores the key but OpenAI SDK requires one

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided website content.

Rules:
1. Answer the question using ONLY the information in the context below.
2. If the context does not contain enough information to answer, say so clearly.
3. Always cite which source (URL) your answer comes from.
4. Be concise and accurate.
5. Do not make up information that is not in the context.
"""


class LLMClient:
    """Wrapper around LM Studio's OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        base_url: str = LM_STUDIO_BASE_URL,
        api_key: str = LM_STUDIO_API_KEY,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = None  # Will be auto-detected from available models

    def _get_model(self) -> str:
        """Auto-detect the loaded model name from LM Studio."""
        if self._model:
            return self._model
        try:
            models = self.client.models.list()
            if models.data:
                self._model = models.data[0].id
                logger.info("Using model: %s", self._model)
                return self._model
        except Exception as exc:
            logger.warning("Could not list models: %s", exc)
        return "local-model"

    def answer(
        self,
        question: str,
        context: str,
        chat_history: List[Dict[str, str]] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate an answer for `question` given the retrieved `context`.

        Parameters
        ----------
        question : str
        context : str — the retrieved chunks joined together
        chat_history : list of {"role": "user"/"assistant", "content": str}
        temperature : float — lower = more factual
        max_tokens : int

        Returns
        -------
        str — the assistant's answer
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Include previous turns for multi-turn conversation
        if chat_history:
            messages.extend(chat_history[-6:])  # Keep last 3 exchanges

        user_message = (
            f"Context from the website:\n\n{context}\n\n"
            f"Question: {question}"
        )
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model=self._get_model(),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            raise ConnectionError(
                f"Could not reach LM Studio at {LM_STUDIO_BASE_URL}. "
                "Make sure LM Studio is running and the local server is started."
            ) from exc

    def stream_answer(
        self,
        question: str,
        context: str,
        chat_history: List[Dict[str, str]] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Generator[str, None, None]:
        """Same as `answer` but streams tokens one by one."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if chat_history:
            messages.extend(chat_history[-6:])

        user_message = (
            f"Context from the website:\n\n{context}\n\n"
            f"Question: {question}"
        )
        messages.append({"role": "user", "content": user_message})

        try:
            stream = self.client.chat.completions.create(
                model=self._get_model(),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            raise ConnectionError(
                f"Could not reach LM Studio at {LM_STUDIO_BASE_URL}."
            ) from exc

    def is_available(self) -> bool:
        """Check if LM Studio server is reachable."""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

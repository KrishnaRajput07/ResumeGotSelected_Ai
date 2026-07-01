# ─────────────────────────────────────────────────────────────────────────────
# Ollama LLM Client — Handles all Qwen3 interactions
# ─────────────────────────────────────────────────────────────────────────────
# Engineering notes:
#   - All structured output calls use format="json" (Ollama native JSON mode)
#   - Tenacity provides retry/backoff for transient failures
#   - Thinking mode is DISABLED for structured calls (speed + determinism)
#   - Thinking mode is ENABLED for complex reasoning tasks if needed
# ─────────────────────────────────────────────────────────────────────────────

import json
import logging
import re
from typing import Any, Optional
import ollama
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from core.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Thin wrapper around Ollama Python client.
    
    Two call modes:
    1. structured_call() → forces JSON output, temperature=0, no thinking
       Used for: JD parsing, criterion scoring, gate checks
       
    2. reasoning_call() → free text, temperature=0.1, thinking optional
       Used for: Q&A answer text, open signal mining (non-JSON parts)
    """

    def __init__(self):
        self.client = ollama.Client(host=settings.ollama_base_url)
        self.model = settings.ollama_model

    def _build_options(self, temperature: float) -> dict:
        return {
            "temperature": temperature,
            "num_predict": 2048,     # Max tokens to generate
            "top_p": 0.9,
        }

    @retry(
        stop=stop_after_attempt(settings.ollama_max_retries),
        wait=wait_exponential(multiplier=1, min=settings.ollama_retry_wait, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def structured_call(
        self,
        system_prompt: str,
        user_prompt: str,
        expected_keys: Optional[list[str]] = None,
    ) -> dict:
        """
        LLM call that MUST return valid JSON.
        Uses Ollama's format='json' to enforce JSON output.
        
        If the model returns invalid JSON after retries, raises ValueError.
        
        Args:
            system_prompt: System-level instruction (schema, rules)
            user_prompt: The actual content to process
            expected_keys: Optional list of top-level keys to validate in response
        
        Returns:
            Parsed dict from JSON output
        """
        # Qwen3: prepend /no_think to skip thinking tokens for speed
        # This is critical for structured calls — thinking tokens pollute JSON output
        full_system = f"/no_think\n\n{system_prompt}"
        
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": full_system},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
            options=self._build_options(settings.llm_temp_structured),
        )
        
        raw_text = response.message.content
        
        # Parse JSON — handle common model quirks
        parsed = self._safe_json_parse(raw_text)
        
        # Validate expected keys if provided
        if expected_keys:
            missing = [k for k in expected_keys if k not in parsed]
            if missing:
                raise ValueError(
                    f"LLM response missing required keys: {missing}. "
                    f"Got: {list(parsed.keys())}"
                )
        
        logger.debug(f"structured_call success | keys={list(parsed.keys())}")
        return parsed

    @retry(
        stop=stop_after_attempt(settings.ollama_max_retries),
        wait=wait_exponential(multiplier=1, min=settings.ollama_retry_wait, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def reasoning_call(
        self,
        system_prompt: str,
        user_prompt: str,
        use_thinking: bool = False,
    ) -> str:
        """
        LLM call that returns free text.
        
        Args:
            system_prompt: System instruction
            user_prompt: Content to process
            use_thinking: If True, enables Qwen3's extended thinking mode
        
        Returns:
            Raw string response (model's answer text)
        """
        prefix = "" if use_thinking else "/no_think\n\n"
        full_system = f"{prefix}{system_prompt}"
        
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": full_system},
                {"role": "user", "content": user_prompt},
            ],
            options=self._build_options(settings.llm_temp_reasoning),
        )
        
        return response.message.content.strip()

    def _safe_json_parse(self, text: str) -> dict:
        """
        Robustly parses JSON from LLM output.
        Handles: markdown fences, leading text, trailing text.
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Strip markdown fences
        fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass
        
        # Find the first { ... } block
        brace_match = re.search(r"\{[\s\S]+\}", text)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass
        
        raise ValueError(
            f"Could not parse JSON from LLM output. "
            f"First 200 chars: {text[:200]!r}"
        )

    def check_connectivity(self) -> bool:
        """Verify Ollama is running and model is available."""
        try:
            models = self.client.list()
            model_names = [m.model for m in models.models]
            available = any(self.model in m for m in model_names)
            if not available:
                logger.warning(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available: {model_names}. "
                    f"Run: ollama pull {self.model}"
                )
            return available
        except Exception as e:
            logger.error(f"Ollama connectivity check failed: {e}")
            return False


# Singleton instance
llm_client = LLMClient()

import logging
from typing import Optional

from google import genai

from config import config

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, model_name: Optional[str] = None):
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        _ = self.client.aio
        self.model_name = model_name or config.GEMINI_MODEL

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        kwargs = {
            "model": self.model_name,
            "input": prompt,
        }
        if system_prompt:
            kwargs["system_instruction"] = system_prompt
        response = self.client.interactions.create(**kwargs)
        return response.output_text

    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        kwargs = {
            "model": self.model_name,
            "input": prompt,
        }
        if system_prompt:
            kwargs["system_instruction"] = system_prompt
        if temperature is not None:
            kwargs["generation_config"] = {"temperature": temperature}
        response = await self.client.aio.interactions.create(**kwargs)
        return response.output_text

    def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        kwargs = {
            "model": self.model_name,
            "input": prompt,
        }
        if system_prompt:
            kwargs["system_instruction"] = system_prompt
        kwargs["generation_config"] = {"response_mime_type": "application/json"}
        response = self.client.interactions.create(**kwargs)
        return response.output_text

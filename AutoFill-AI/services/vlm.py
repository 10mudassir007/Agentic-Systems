import base64
import logging
import re
from typing import Optional

from google import genai

from config import config

logger = logging.getLogger(__name__)


class VLMService:
    """Vision-Language Model service for screenshot-based element detection.

    Uses Gemini Vision to find form fields and buttons on a page screenshot.
    Serves as a fallback when DOM-based selectors fail.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model_name = model_name or config.GEMINI_MODEL

    async def locate_field(
        self, screenshot_bytes: bytes, field_label: str, field_type: str = "short_text"
    ) -> Optional[str]:
        """Use VLM to find a CSS selector for a form field by its label."""
        b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        prompt = (
            f"Find the form field labeled '{field_label}' (type: {field_type}) "
            "in this screenshot. Return ONLY a CSS selector string that Playwright "
            "can use to locate this element. The selector must uniquely identify "
            "the input element for this field. If the field has an id, return '#id'. "
            "If it has a name, return '[name=\"...\"]'. Otherwise, use an attribute "
            "selector, aria-label, or nth-child that uniquely identifies the element. "
            "Return ONLY the selector string, nothing else."
        )

        try:
            response = await self.client.aio.interactions.create(
                model=self.model_name,
                input=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "data": b64,
                        "mime_type": "image/png",
                    },
                ],
            )
            selector = response.output_text.strip()
            if selector:
                selector = selector.strip("`").strip()
                logger.info("VLM located field '%s' → %s", field_label, selector)
                return selector
        except Exception as exc:
            logger.warning("VLM locate_field failed for '%s': %s", field_label, exc)

        return None

    async def locate_submit_button(self, screenshot_bytes: bytes) -> Optional[str]:
        """Use VLM to find the submit button on a form page."""
        b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        prompt = (
            "Find the submit button on this form page. Look for buttons or links "
            "with text like 'Submit', 'Send', 'Submit Form', 'Next', 'Continue', "
            "or the submit/next action button. Return ONLY a CSS selector string "
            "that Playwright can use to click this button. The selector must uniquely "
            "identify the element. Return ONLY the selector string, nothing else."
        )

        try:
            response = await self.client.aio.interactions.create(
                model=self.model_name,
                input=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "data": b64,
                        "mime_type": "image/png",
                    },
                ],
            )
            selector = response.output_text.strip()
            if selector:
                selector = selector.strip("`").strip()
                logger.info("VLM located submit button → %s", selector)
                return selector
        except Exception as exc:
            logger.warning("VLM locate_submit_button failed: %s", exc)

        return None

    async def detect_all_fields(
        self, screenshot_bytes: bytes
    ) -> Optional[dict]:
        """Use VLM to get a complete mapping of field labels to selectors on the page."""
        b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        prompt = (
            "Analyze this form screenshot. For EVERY form field visible on the page, "
            "return a JSON object mapping field labels to CSS selectors. "
            "Each key should be the field's label text (e.g. 'First Name', 'Email'). "
            "Each value should be a CSS selector that Playwright can use to locate "
            "the input element for that field. "
            "Return ONLY valid JSON, no other text. Example:\n"
            '{"First Name": "#firstName", "Email": "[name=\\"email\\"]"}'
        )

        try:
            response = await self.client.aio.interactions.create(
                model=self.model_name,
                input=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "data": b64,
                        "mime_type": "image/png",
                    },
                ],
            )
            text = response.output_text.strip()
            text = text.strip("`").strip()
            if text.startswith("json"):
                text = text[4:].strip()
            import json

            parsed = json.loads(text)
            if isinstance(parsed, dict):
                logger.info("VLM detected %d fields from screenshot", len(parsed))
                return parsed
        except Exception as exc:
            logger.warning("VLM detect_all_fields failed: %s", exc)

        return None

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from graph.state import GraphState, RunStatus
from services.browser import BrowserManager
from services.vlm import VLMService
from run_registry import registry

logger = logging.getLogger(__name__)


class SubmitterAgent:
    """Waits after form filling, then locates and clicks the submit button.

    Uses VLM (screenshot + Gemini Vision) as the primary method to find
    the submit button, with DOM-based fallback selectors.
    """

    SUBMIT_SELECTORS = [
        'button[type="submit"]',
        'input[type="submit"]',
        'button:has-text("Submit")',
        'button:has-text("Send")',
        'button:has-text("Next")',
        'button:has-text("Continue")',
        'button:has-text("Done")',
        '[role="button"]:has-text("Submit")',
        'a:has-text("Submit")',
        '.freebirdFormviewerViewNavigationSubmitButton',
        '[jsname] button:has-text("Submit")',
        '[jscontroller] button:has-text("Submit")',
    ]

    def __init__(self) -> None:
        self.browser = BrowserManager()
        self.vlm = VLMService()

    async def submit(self, state: GraphState) -> Dict[str, Any]:
        run_id = state["run_id"]
        logger.info("Submitter: waiting 10s before submitting...")

        await registry.publish_status(run_id, stage="submitter", status="running")

        await asyncio.sleep(10)

        try:
            page = await self.browser.get_page(run_id)
            screenshot = await page.screenshot(full_page=True)

            selector = await self.vlm.locate_submit_button(screenshot)

            if not selector or not await self._try_selector(page, selector):
                logger.info("VLM selector failed, trying DOM fallbacks...")
                selector = await self._try_dom_fallbacks(page)

            if selector:
                await page.click(selector)
                logger.info("Submitter: clicked submit button → %s", selector)

                await page.wait_for_timeout(3000)

                confirmation_text = "Your response has been recorded."
                try:
                    confirmation = page.get_by_text(confirmation_text, exact=False)
                    if await confirmation.is_visible(timeout=5000):
                        logger.info("Confirmation message detected, closing browser")
                        await registry.publish_status(run_id, stage="submitter", status="done", confirmed=True)
                        await self.browser.close_page(run_id)
                        return {
                            "submitted": True,
                            "status": RunStatus.COMPLETED,
                            "next_agent": "__end__",
                            "completion_message": "Form submitted and confirmed.",
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                        }
                except Exception:
                    pass

                await registry.publish_status(run_id, stage="submitter", status="done")
                return {
                    "submitted": True,
                    "status": RunStatus.COMPLETED,
                    "next_agent": "__end__",
                    "completion_message": "Form submitted successfully.",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }

            await registry.publish_status(run_id, stage="submitter", status="done", note="no_submit_button")
            return {
                "submitted": True,
                "status": RunStatus.COMPLETED,
                "next_agent": "__end__",
                "completion_message": "Form filled (no submit button found).",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as exc:
            logger.exception("Submitter failed")
            await registry.publish_status(run_id, stage="submitter", status="error", error_message=str(exc))
            return {
                "submitted": True,
                "status": RunStatus.COMPLETED,
                "next_agent": "__end__",
                "completion_message": f"Submit failed: {exc}",
                "error_message": f"Submit failed: {exc}",
            }

    async def _try_selector(self, page, selector: str) -> bool:
        try:
            el = await page.query_selector(selector)
            if el and await el.is_visible():
                return True
        except Exception:
            pass
        return False

    async def _try_dom_fallbacks(self, page) -> str | None:
        for sel in self.SUBMIT_SELECTORS:
            try:
                el = await page.query_selector(sel)
                if el and await el.is_visible():
                    logger.info("DOM fallback found submit button → %s", sel)
                    return sel
            except Exception:
                continue

        try:
            buttons = await page.query_selector_all(
                'button, input[type="submit"], [role="button"]'
            )
            for btn in buttons:
                text = await btn.inner_text()
                if text and any(
                    kw in text.lower()
                    for kw in ["submit", "send", "next", "continue", "done", "ok"]
                ):
                    if await btn.is_visible():
                        tag = await btn.get_attribute("id") or ""
                        if tag:
                            return f"#{tag}"
                        return await btn.get_attribute("data-field-id") or (
                            f"button:has-text(\"{text.strip()}\")"
                        )
        except Exception:
            pass

        return None

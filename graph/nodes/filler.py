from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from graph.state import GraphState, RunStatus, FieldStatus, FieldResult
from services.browser import BrowserManager
from services.vlm import VLMService
from run_registry import registry

logger = logging.getLogger(__name__)


class FillerAgent:
    """Types resolved values into form fields via Playwright."""

    def __init__(self) -> None:
        self.browser = BrowserManager()
        self.vlm = VLMService()

    async def fill_fields(self, state: GraphState) -> Dict[str, Any]:
        run_id = state["run_id"]
        resolved = state.get("resolved_fields", {})

        await registry.publish_status(run_id, stage="filler", status="running", fields_to_fill=len(resolved))

        if not resolved:
            await registry.publish_status(run_id, stage="filler", status="error", error_message="No resolved fields to fill.")
            return {
                "status": RunStatus.FAILED,
                "error_message": "No resolved fields to fill.",
                "next_agent": "__end__",
            }

        page = await self.browser.get_page(run_id)
        prev_results: Dict[str, FieldResult] = dict(state.get("filler_results", {}))
        errors: List[str] = []

        for field_id, result in resolved.items():
            if field_id in prev_results:
                continue
            try:
                ok = await self._fill_one(page, field_id, result, state.get("fields", []))
                if ok:
                    prev_results[field_id] = FieldResult(
                        field_id=field_id,
                        status=FieldStatus.FILLED,
                        value=result["value"],
                        confidence=None,
                        error_message=None,
                        human_question=None,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                else:
                    prev_results[field_id] = FieldResult(
                        field_id=field_id,
                        status=FieldStatus.FILL_ERROR,
                        value=result["value"],
                        confidence=None,
                        error_message="Failed to fill field",
                        human_question=None,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    errors.append(field_id)
            except Exception as exc:
                logger.exception("Filler error on %s", field_id)
                prev_results[field_id] = FieldResult(
                    field_id=field_id,
                    status=FieldStatus.FILL_ERROR,
                    value=result["value"],
                    confidence=None,
                    error_message=str(exc),
                    human_question=None,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                errors.append(field_id)

        next_agent = "supervisor" if errors else "__end__"

        status_tag = "error" if errors else "done"
        await registry.publish_status(
            run_id, stage="filler", status=status_tag, filled=len(prev_results) - len(errors), errors=len(errors)
        )

        return {
            "filler_results": prev_results,
            "filler_errors": errors,
            "status": RunStatus.FILLING_FORM if errors else RunStatus.COMPLETED,
            "next_agent": next_agent,
            "completion_message": None if errors else "Form filled successfully.",
        }

    async def _fill_one(
        self, page, field_id: str, result: FieldResult, fields: List[Dict[str, Any]]
    ) -> bool:
        field = next((f for f in fields if f["field_id"] == field_id), None)
        if field is None:
            logger.warning("Field %s not found in field list", field_id)
            return False

        selector = field.get("selector")
        value = result.get("value")
        if not selector or not value:
            return False

        el = None
        try:
            el = await page.wait_for_selector(selector, timeout=3000)
        except Exception:
            pass

        # VLM fallback: take screenshot and ask Gemini where the field is
        if el is None:
            logger.info("DOM selector failed for '%s', trying VLM...", field.get("field_label"))
            try:
                screenshot = await page.screenshot(full_page=True)
                vlm_selector = await self.vlm.locate_field(
                    screenshot,
                    field.get("field_label", ""),
                    field.get("field_type", "short_text"),
                )
                if vlm_selector:
                    try:
                        el = await page.wait_for_selector(vlm_selector, timeout=3000)
                    except Exception:
                        el = None
            except Exception as exc:
                logger.warning("VLM fallback failed for '%s': %s", field.get("field_label"), exc)

        if el is None:
            return False

        ftype = field.get("field_type", "short_text")

        if ftype in ("dropdown",):
            await el.select_option(value)
        elif ftype in ("checkbox", "radio"):
            if str(value).lower() in ("true", "yes", "on", "1"):
                await el.check()
            else:
                await el.uncheck()
        else:
            await el.click()
            await el.fill("")
            await el.fill(value)

        await page.evaluate(
            "(el) => el.dispatchEvent(new Event('change', { bubbles: true }))", el
        )
        return True

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import config
from graph.state import GraphState, RunStatus, FieldStatus, FieldResult, Field
from services.llm import LLMClient
from run_registry import registry

logger = logging.getLogger(__name__)

MAPPING_SYSTEM_PROMPT = """\
You are a data mapping assistant. Your job is to match form field labels to the \
best value from a person's bio data.

Rules:
1. Use semantic matching, not exact string matching. "Full Name" and "Legal Name" \
should both resolve to the person's full name.
2. If the field asks for something that does not exist in the bio data, return \
"value": null with a low confidence score.
3. For multiple-choice or checkbox fields, pick from the provided options list if \
one matches. If none match, return null.
4. Be conservative — if you are unsure, give a low confidence score.

Respond with a JSON object:
{{
  "value": "<the mapped value or null>",
  "confidence": <0.0-1.0>,
  "explanation": "<brief reasoning>"
}}

Bio data:
{bio}
"""


class DataMapperAgent:
    """Maps form fields to bio data values using Gemini.

    This is the **only** agent that loads bio data.  It reads it from
    config.BIO_DATA (originally parsed from my_data.md) and includes it
    in the LLM system prompt.  Bio data never enters the shared graph state.
    """

    def __init__(self) -> None:
        self.llm = LLMClient()
        self.threshold = config.CONFIDENCE_THRESHOLD
        self.batch_size = config.BATCH_SIZE

    async def map_fields(self, state: GraphState) -> Dict[str, Any]:
        run_id = state["run_id"]
        fields = state.get("fields", [])

        await registry.publish_status(run_id, stage="data_mapper", status="running", field_count=len(fields))

        if not fields:
            await registry.publish_status(run_id, stage="data_mapper", status="error", error_message="No fields to map.")
            return {
                "status": RunStatus.FAILED,
                "error_message": "No fields to map.",
                "next_agent": "__end__",
            }

        resolved: Dict[str, FieldResult] = dict(state["resolved_fields"])
        unresolved: Dict[str, FieldResult] = dict(state["unresolved_fields"])

        prev_human_answers = state.get("human_answers", {})

        to_process = [
            f
            for f in fields
            if f["field_id"] not in resolved and f["field_id"] not in unresolved
        ]

        if not to_process:
            next_agent = "filler" if not unresolved else "supervisor"
            await registry.publish_status(run_id, stage="data_mapper", status="done")
            return {
                "resolved_fields": resolved,
                "unresolved_fields": unresolved,
                "status": RunStatus.FILLING_FORM if not unresolved else RunStatus.MAPPING_FIELDS,
                "next_agent": next_agent,
            }

        # Process in batches
        for i in range(0, len(to_process), self.batch_size):
            batch = to_process[i : i + self.batch_size]
            results = await self._map_batch(batch, prev_human_answers)

            for r in results:
                key = r["field_id"]
                if r["status"] == FieldStatus.RESOLVED:
                    resolved[key] = r
                else:
                    unresolved[key] = r

        next_agent = "filler" if not unresolved else "supervisor"

        await registry.publish_status(
            run_id,
            stage="data_mapper",
            status="done",
            resolved=len(resolved),
            unresolved=len(unresolved),
        )

        return {
            "resolved_fields": resolved,
            "unresolved_fields": unresolved,
            "status": RunStatus.FILLING_FORM if not unresolved else RunStatus.MAPPING_FIELDS,
            "next_agent": next_agent,
        }

    async def _map_batch(
        self, fields: List[Field], human_answers: Dict[str, str]
    ) -> List[FieldResult]:
        results: List[FieldResult] = []
        for field in fields:
            # If a human already answered this field, skip LLM
            if field["field_id"] in human_answers:
                results.append(
                    FieldResult(
                        field_id=field["field_id"],
                        status=FieldStatus.RESOLVED,
                        value=human_answers[field["field_id"]],
                        confidence=1.0,
                        error_message=None,
                        human_question=None,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                )
                continue
            result = await self._map_single(field)
            results.append(result)
        return results

    async def _map_single(self, field: Field) -> FieldResult:
        prompt = self._build_prompt(field)
        system = MAPPING_SYSTEM_PROMPT.format(
            bio=config.BIO_DATA_PROMPT
        )

        try:
            raw = await self.llm.generate_async(prompt, system_prompt=system, temperature=0.2)
            parsed = self._parse_llm_json(raw)
            if parsed is None:
                logger.warning("Failed to parse LLM response for %s", field["field_label"])
                return self._unresolved(field, 0.0, "LLM response could not be parsed")

            value = parsed.get("value")
            confidence = float(parsed.get("confidence", 0.0))

            if value is not None and confidence >= self.threshold:
                return FieldResult(
                    field_id=field["field_id"],
                    status=FieldStatus.RESOLVED,
                    value=str(value),
                    confidence=confidence,
                    error_message=None,
                    human_question=None,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

            return self._unresolved(
                field,
                confidence,
                parsed.get("explanation", "Low confidence or missing value"),
            )

        except Exception as exc:
            logger.error("LLM call failed for %s: %s", field["field_label"], exc)
            return self._unresolved(field, 0.0, str(exc))

    def _build_prompt(self, field: Field) -> str:
        lines = ["Map this form field to a value from the bio data above.", ""]
        lines.append(f"Label:       {field['field_label']}")
        lines.append(f"Type:        {field['field_type']}")
        lines.append(f"Required:    {field['is_required']}")

        if field.get("options"):
            lines.append(f"Options:     {', '.join(field['options'])}")

        if field.get("placeholder"):
            lines.append(f"Placeholder: {field['placeholder']}")

        lines.append("")
        lines.append("Respond with the JSON object only.")
        return "\n".join(lines)

    @staticmethod
    def _parse_llm_json(raw: str) -> Optional[Dict[str, Any]]:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _unresolved(field: Field, confidence: float, reason: str) -> FieldResult:
        return FieldResult(
            field_id=field["field_id"],
            status=FieldStatus.UNRESOLVED,
            value=None,
            confidence=confidence,
            error_message=None,
            human_question=f"Please provide a value for: {field['field_label']}",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from langgraph.types import interrupt

from graph.state import GraphState, RunStatus, FieldStatus
from run_registry import registry

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """Hub-and-spoke orchestrator.

    Every agent node routes back here.  Supervisor inspects the
    current graph state and decides what happens next:

        form_reader  ──►  supervisor  ──►  data_mapper  ──►  supervisor
                                          filler       ──►  supervisor
                                          human interrupt    supervisor
                                          __end__

    This agent does **not** touch bio data.  That boundary is
    enforced by keeping bio data out of GraphState entirely.
    """

    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public entry point — called by LangGraph as the "supervisor" node
    # ------------------------------------------------------------------

    async def route(self, state: GraphState) -> Dict[str, Any]:
        """Single decision point.  Returns state patches + next_agent."""
        run_id = state["run_id"]
        logger.info(
            "Supervisor[%s] status=%s next=%s",
            run_id[:8],
            state["status"].value,
            state.get("next_agent"),
        )

        # 0. Fatal failure — stop (preserve FAILED status)
        if state["status"] == RunStatus.FAILED:
            await registry.publish_status(run_id, stage="supervisor", status="failed", error_message=state.get("error_message"))
            return {
                "next_agent": "__end__",
                "completion_message": None,
                "error_message": state.get("error_message") or "Run failed",
            }

        # 1. Human interrupt — execute the actual interrupt() call
        if state["awaiting_human"]:
            return await self._execute_human_interrupt(state)

        # 2. Human interrupt requested → first, flag the state
        if self._should_pause_for_human(state):
            return await self._prepare_human_interrupt(state)

        # 2. Form not yet read → form_reader (only if not already failed)
        if not state["fields_loaded"]:
            if state.get("form_reader_attempts", 0) >= 1:
                await registry.publish_status(run_id, stage="supervisor", status="failed", error_message="Form reader failed after 1 attempt")
                return {
                    "status": RunStatus.FAILED,
                    "next_agent": "__end__",
                    "error_message": "Form reader failed after 1 attempt",
                }
            return {
                **self._goto("form_reader", RunStatus.READING_FORM),
                "form_reader_attempts": state.get("form_reader_attempts", 0) + 1,
            }

        # 3. Fields loaded but never mapped → data_mapper
        if not self._has_mapping(state):
            return self._goto("data_mapper", RunStatus.MAPPING_FIELDS)

        # 4. Data mapper reported unresolved fields → retry or human
        if state["unresolved_fields"]:
            return await self._handle_unresolved(state)

        # 5. Filler reported errors → retry or human
        if state["filler_errors"]:
            return await self._handle_filler_errors(state)

        # 6. Resolved fields exist but not yet filled → filler
        if state["resolved_fields"] and not state["filler_results"]:
            return self._goto("filler", RunStatus.FILLING_FORM)

        # 7. Some fields filled, some still pending → filler
        if self._pending_fills(state):
            return self._goto("filler", RunStatus.FILLING_FORM)

        # 8. All fields filled → submit via VLM
        if self._all_done(state) and not state.get("submitted"):
            return {
                "status": RunStatus.FILLING_FORM,
                "next_agent": "submitter",
            }

        # 9. Everything done (submitted or not needed) → end
        if self._all_done(state):
            return self._complete(state)

        # 10. Safety net — nothing left to do
        logger.warning("Supervisor reached fallthrough — ending run")
        return self._end()

    # ------------------------------------------------------------------
    # Internal routing helpers
    # ------------------------------------------------------------------

    def _should_pause_for_human(self, state: GraphState) -> bool:
        if state["awaiting_human"]:
            return True
        for fid in state["unresolved_fields"]:
            if state["field_retry_counts"].get(fid, 0) >= self.max_retries:
                return True
        return False

    def _has_mapping(self, state: GraphState) -> bool:
        return bool(state["resolved_fields"]) or bool(state["unresolved_fields"])

    def _pending_fills(self, state: GraphState) -> bool:
        resolved = set(state["resolved_fields"].keys())
        filled = set(state["filler_results"].keys())
        return bool(resolved - filled)

    def _all_done(self, state: GraphState) -> bool:
        if not state["resolved_fields"]:
            return False
        all_filled = all(
            fid in state["filler_results"] for fid in state["resolved_fields"]
        )
        has_errors = any(
            r["status"] == FieldStatus.FILL_ERROR
            for r in state["filler_results"].values()
        )
        return all_filled and not has_errors

    @staticmethod
    def _goto(agent: str, status: RunStatus) -> Dict[str, Any]:
        return {
            "status": status,
            "next_agent": agent,
            "error_message": None,
        }

    # ------------------------------------------------------------------
    # Unresolved / retry logic
    # ------------------------------------------------------------------

    async def _handle_unresolved(self, state: GraphState) -> Dict[str, Any]:
        run_id = state["run_id"]
        retry = dict(state["field_retry_counts"])
        keep: Dict[str, Any] = {}
        mapping_attempts = state.get("mapping_attempts", 0) + 1

        # Hard safety cap on total mapping cycles
        if mapping_attempts > 5:
            await registry.publish_status(run_id, stage="supervisor", status="failed", error_message="Too many mapping attempts")
            return {
                "status": RunStatus.FAILED,
                "next_agent": "__end__",
                "error_message": "Too many mapping attempts — giving up",
            }

        for fid, result in state["unresolved_fields"].items():
            if retry.get(fid, 0) >= self.max_retries:
                return await self._prepare_human_interrupt(state)
            retry[fid] = retry.get(fid, 0) + 1
            keep[fid] = result

        await registry.publish_status(run_id, stage="supervisor", status="running", substage="retry_mapping")
        return {
            "field_retry_counts": retry,
            "unresolved_fields": keep,
            "mapping_attempts": mapping_attempts,
            "status": RunStatus.MAPPING_FIELDS,
            "next_agent": "data_mapper",
        }

    async def _handle_filler_errors(self, state: GraphState) -> Dict[str, Any]:
        run_id = state["run_id"]
        retry = dict(state["field_retry_counts"])
        for fid in state["filler_errors"]:
            if retry.get(fid, 0) >= self.max_retries:
                return await self._prepare_human_interrupt(state)

        for fid in state["filler_errors"]:
            retry[fid] = retry.get(fid, 0) + 1

        await registry.publish_status(run_id, stage="supervisor", status="running", substage="retry_fill")
        return {
            "filler_errors": [],
            "filler_results": {},
            "field_retry_counts": retry,
            "status": RunStatus.FILLING_FORM,
            "next_agent": "filler",
        }

    # ------------------------------------------------------------------
    # Human-in-the-loop
    # ------------------------------------------------------------------

    async def _prepare_human_interrupt(self, state: GraphState) -> Dict[str, Any]:
        run_id = state["run_id"]
        questions = self._build_questions(state)
        logger.info("Supervisor requesting human input (%d questions)", len(questions))

        await registry.publish_status(
            run_id,
            stage="supervisor",
            status="waiting_for_human",
            human_questions=[q["question"] for q in questions],
        )

        return {
            "awaiting_human": True,
            "status": RunStatus.WAITING_FOR_HUMAN,
            "next_agent": "supervisor",
            "human_questions": [q["question"] for q in questions],
        }

    async def _execute_human_interrupt(self, state: GraphState) -> Dict[str, Any]:
        run_id = state["run_id"]
        questions = self._build_questions(state)
        payload = {
            "run_id": run_id,
            "questions": questions,
            "type": "human_input_request",
        }

        logger.info("Supervisor executing human interrupt (%d questions)", len(questions))

        human_response = interrupt(payload)

        if isinstance(human_response, dict) and "answers" in human_response:
            return self._apply_human_answers(state, human_response["answers"])

        return {
            "awaiting_human": True,
            "status": RunStatus.WAITING_FOR_HUMAN,
            "next_agent": "supervisor",
        }

    def _build_questions(self, state: GraphState) -> List[Dict[str, Any]]:
        field_map = {f["field_id"]: f for f in state["fields"]}
        questions: List[Dict[str, Any]] = []
        for fid, result in state["unresolved_fields"].items():
            f = field_map.get(fid, {})
            questions.append(
                {
                    "field_id": fid,
                    "question": f.get("field_label", "Unknown field"),
                    "field_type": f.get("field_type", "short_text"),
                    "is_required": f.get("is_required", True),
                    "options": f.get("options"),
                    "confidence": result.get("confidence"),
                    "error_message": result.get("error_message"),
                }
            )
        return questions

    def _apply_human_answers(
        self, state: GraphState, answers: Dict[str, str]
    ) -> Dict[str, Any]:
        unresolved = dict(state["unresolved_fields"])
        resolved = dict(state["resolved_fields"])

        # Build label→field_id lookup so answers keyed by field_label still work
        label_to_fid: Dict[str, str] = {}
        for f in state["fields"]:
            label_to_fid[f["field_label"]] = f["field_id"]

        for key, answer in answers.items():
            fid = label_to_fid.get(key, key)
            if fid not in unresolved:
                continue
            result = dict(unresolved[fid])
            result["value"] = answer
            result["status"] = FieldStatus.RESOLVED
            result["confidence"] = 1.0
            resolved[fid] = result
            del unresolved[fid]

        logger.info("Human answered %d fields", len(answers))

        return {
            "unresolved_fields": unresolved,
            "resolved_fields": resolved,
            "awaiting_human": False,
            "human_questions": [],
            "status": RunStatus.MAPPING_FIELDS,
            "next_agent": "data_mapper",
        }

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    @staticmethod
    def _complete(state: GraphState) -> Dict[str, Any]:
        return {
            "status": RunStatus.COMPLETED,
            "next_agent": "__end__",
            "completion_message": "Form filled successfully.",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error_message": None,
        }

    @staticmethod
    def _end() -> Dict[str, Any]:
        return {
            "status": RunStatus.COMPLETED,
            "next_agent": "__end__",
            "completion_message": "No more work to do.",
            "error_message": None,
        }

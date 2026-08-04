from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from graph.state import GraphState, RunStatus, make_initial_state
from graph.nodes.supervisor import SupervisorAgent
from graph.nodes.form_reader import FormReaderAgent
from graph.nodes.data_mapper import DataMapperAgent
from graph.nodes.filler import FillerAgent
from graph.nodes.submitter import SubmitterAgent
from run_registry import registry

logger = logging.getLogger(__name__)


class AutoFillGraph:
    """Hub-and-spoke LangGraph pipeline.

    Every agent node returns to the Supervisor, which decides the
    next step (or ends the run).
    """

    def __init__(
        self,
        max_retries: int = 3,
    ) -> None:
        self.supervisor = SupervisorAgent(max_retries=max_retries)
        self.form_reader = FormReaderAgent()
        self.data_mapper = DataMapperAgent()
        self.filler = FillerAgent()
        self.submitter = SubmitterAgent()
        self._graph = self._build()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build(self) -> StateGraph:
        builder = StateGraph(GraphState)

        builder.add_node("supervisor", self.supervisor.route)
        builder.add_node("form_reader", self.form_reader.read_form)
        builder.add_node("data_mapper", self.data_mapper.map_fields)
        builder.add_node("filler", self.filler.fill_fields)
        builder.add_node("submitter", self.submitter.submit)

        builder.set_entry_point("supervisor")

        # Hub-and-spoke: supervisor dispatches to any agent
        builder.add_conditional_edges(
            "supervisor",
            _next_agent,
            {
                "form_reader": "form_reader",
                "data_mapper": "data_mapper",
                "filler": "filler",
                "submitter": "submitter",
                "supervisor": "supervisor",
                "__end__": END,
            },
        )

        # Every agent routes back to supervisor
        builder.add_edge("form_reader", "supervisor")
        builder.add_edge("data_mapper", "supervisor")
        builder.add_edge("filler", "supervisor")
        builder.add_edge("submitter", "supervisor")

        return builder.compile(checkpointer=MemorySaver())

    # ------------------------------------------------------------------
    # Background execution (full lifecycle with human-in-the-loop)
    # ------------------------------------------------------------------

    async def execute_run(self, run_id: str, form_url: str) -> dict:
        """Run the full graph lifecycle in a single background task.

        Handles human-in-the-loop pauses/resumes automatically by
        waiting on the run's resume_event from the registry.
        Returns the final state dict.
        """
        from langgraph.types import Command

        initial = make_initial_state(run_id, form_url)
        config = {"configurable": {"thread_id": run_id}}

        # --- first pass: start from initial state ---
        final = await self._stream(initial, config)

        # --- human-in-the-loop loop ---
        while final and final.get("awaiting_human"):
            run_state = registry.get(run_id)
            if not run_state:
                break

            await registry.publish_status(
                run_id,
                stage="supervisor",
                status="waiting_for_human",
                human_questions=final.get("human_questions", []),
            )

            await run_state.resume_event.wait()
            run_state.resume_event.clear()
            answers = dict(run_state.resume_answers)
            run_state.resume_answers.clear()

            await registry.publish_status(run_id, stage="supervisor", status="resumed")

            final = await self._stream(
                Command(resume={"answers": answers}),
                config,
            )

        return final

    async def _stream(self, input_data, config: dict) -> dict:
        """Iterate the graph stream, return the last event."""
        final = {}
        async for event in self._graph.astream(input_data, config, stream_mode="values"):
            final = event
        return final


def _next_agent(state: GraphState) -> str:
    """Conditional edge router — returns the next agent name or __end__."""
    return state.get("next_agent") or "__end__"

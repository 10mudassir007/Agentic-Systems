from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class RunState:
    def __init__(self, run_id: str, form_url: str) -> None:
        self.run_id = run_id
        self.form_url = form_url
        self.status_queues: List[asyncio.Queue] = []
        self.frame_queues: List[asyncio.Queue] = []
        self.latest_status: Dict[str, Any] = {
            "run_id": run_id,
            "form_url": form_url,
            "stage": "init",
            "status": "created",
        }
        self.resume_event = asyncio.Event()
        self.resume_answers: Dict[str, str] = {}


class RunRegistry:
    def __init__(self) -> None:
        self._runs: Dict[str, RunState] = {}

    def create(self, run_id: str, form_url: str) -> RunState:
        state = RunState(run_id, form_url)
        self._runs[run_id] = state
        return state

    def get(self, run_id: str) -> RunState | None:
        return self._runs.get(run_id)

    def list_runs(self) -> List[Dict[str, Any]]:
        return [
            {
                "run_id": rs.run_id,
                "form_url": rs.form_url,
                "status": rs.latest_status.get("status", "unknown"),
                "stage": rs.latest_status.get("stage", ""),
            }
            for rs in self._runs.values()
        ]

    def remove(self, run_id: str) -> None:
        self._runs.pop(run_id, None)

    async def publish_status(self, run_id: str, **event: Any) -> None:
        run = self._runs.get(run_id)
        if not run:
            return
        run.latest_status.update(event)
        run.latest_status["run_id"] = run_id
        for q in run.status_queues:
            await q.put(dict(run.latest_status))

    async def publish_frame(self, run_id: str, frame_b64: str) -> None:
        run = self._runs.get(run_id)
        if not run:
            return
        dead: List[asyncio.Queue] = []
        for q in run.frame_queues:
            try:
                q.put_nowait(frame_b64)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(frame_b64)
                except Exception:
                    dead.append(q)
            except Exception:
                dead.append(q)
        for q in dead:
            if q in run.frame_queues:
                run.frame_queues.remove(q)


registry = RunRegistry()

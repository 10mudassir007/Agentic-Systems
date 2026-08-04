from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import config
from graph.builder import AutoFillGraph
from run_registry import registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

graph: AutoFillGraph | None = None
_background_tasks: Dict[str, asyncio.Task] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    graph = AutoFillGraph(max_retries=config.MAX_RETRIES_PER_FIELD)
    logger.info("AutoFill graph ready — model=%s", config.GEMINI_MODEL)
    yield
    # Cancel any remaining background tasks on shutdown
    for rid, task in list(_background_tasks.items()):
        task.cancel()
    _background_tasks.clear()


app = FastAPI(title="AutoFill AI", lifespan=lifespan)
app.frontend("/", directory="dist")
# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    form_url: str


class AnswerRequest(BaseModel):
    answers: dict[str, str]


class BioRequest(BaseModel):
    data: dict[str, Any]


class RunResponse(BaseModel):
    run_id: str
    status: str
    form_url: str
    completion_message: str | None = None
    error_message: str | None = None
    awaiting_human: bool = False
    human_questions: list[str] = []


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------


async def _execute_graph_task(run_id: str, form_url: str) -> None:
    """Background task that runs the full graph lifecycle."""
    try:
        await registry.publish_status(
            run_id, stage="init", status="running", form_url=form_url
        )

        final = await graph.execute_run(run_id, form_url)

        status_val = final.get("status")
        status_str = status_val.value if hasattr(status_val, "value") else str(status_val)

        if final.get("error_message"):
            await registry.publish_status(
                run_id, status="error", error_message=final["error_message"]
            )
        elif status_str in ("completed",):
            await registry.publish_status(
                run_id,
                status="completed",
                completion_message=final.get("completion_message", "Done"),
            )
        else:
            await registry.publish_status(run_id, status=status_str)

    except asyncio.CancelledError:
        logger.info("Background task cancelled for run %s", run_id)
        await registry.publish_status(run_id, status="cancelled")
    except Exception as exc:
        logger.exception("Background task failed for run %s", run_id)
        await registry.publish_status(
            run_id, status="error", error_message=str(exc)
        )
    finally:
        _background_tasks.pop(run_id, None)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/runs", response_model=RunResponse)
async def start_run(body: RunRequest) -> RunResponse:
    run_id = uuid.uuid4().hex[:12]
    logger.info("Starting run %s for %s", run_id, body.form_url)

    registry.create(run_id, body.form_url)

    task = asyncio.create_task(_execute_graph_task(run_id, body.form_url))
    _background_tasks[run_id] = task

    return RunResponse(
        run_id=run_id,
        status="running",
        form_url=body.form_url,
    )


@app.get("/runs")
async def list_runs() -> list[dict]:
    return registry.list_runs()


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    run_state = registry.get(run_id)
    if not run_state:
        raise HTTPException(status_code=404, detail="Run not found")
    return dict(run_state.latest_status)


@app.post("/runs/{run_id}/answer", response_model=RunResponse)
async def submit_answer(run_id: str, body: AnswerRequest) -> RunResponse:
    logger.info("Resuming run %s with %d answers", run_id, len(body.answers))

    run_state = registry.get(run_id)
    if not run_state:
        raise HTTPException(status_code=404, detail="Run not found")

    run_state.resume_answers = dict(body.answers)
    run_state.resume_event.set()

    return RunResponse(
        run_id=run_id,
        status="running",
        form_url=run_state.form_url,
    )


# ---------------------------------------------------------------------------
# Status WebSocket
# ---------------------------------------------------------------------------


@app.websocket("/runs/{run_id}/status/ws")
async def status_websocket(ws: WebSocket, run_id: str) -> None:
    run_state = registry.get(run_id)
    if not run_state:
        await ws.close(code=4004, reason="Run not found")
        return

    await ws.accept()

    # Send current state immediately
    await ws.send_json(run_state.latest_status)

    queue: asyncio.Queue = asyncio.Queue()
    run_state.status_queues.append(queue)
    logger.info("Status WS connected for run %s (%d subscribers)", run_id, len(run_state.status_queues))

    try:
        while True:
            event = await queue.get()
            await ws.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        if queue in run_state.status_queues:
            run_state.status_queues.remove(queue)
        logger.info("Status WS disconnected for run %s (%d subscribers)", run_id, len(run_state.status_queues))


# ---------------------------------------------------------------------------
# Video WebSocket (CDP screencast)
# ---------------------------------------------------------------------------


@app.websocket("/runs/{run_id}/video/ws")
async def video_websocket(ws: WebSocket, run_id: str) -> None:
    run_state = registry.get(run_id)
    if not run_state:
        await ws.close(code=4004, reason="Run not found")
        return

    await ws.accept()

    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    run_state.frame_queues.append(queue)
    logger.info("Video WS connected for run %s (%d subscribers)", run_id, len(run_state.frame_queues))

    try:
        while True:
            frame = await queue.get()
            await ws.send_text(frame)
    except WebSocketDisconnect:
        pass
    finally:
        if queue in run_state.frame_queues:
            run_state.frame_queues.remove(queue)
        logger.info("Video WS disconnected for run %s (%d subscribers)", run_id, len(run_state.frame_queues))


# ---------------------------------------------------------------------------
# Bio / Health (unchanged)
# ---------------------------------------------------------------------------


@app.post("/bio")
async def set_bio(body: BioRequest) -> dict:
    config.update_bio_data(body.data)
    logger.info("Bio data updated (%d sections)", len(body.data))
    return {"status": "ok", "sections": len(body.data)}


@app.get("/bio")
async def get_bio() -> dict:
    return {
        "sections": len(config.BIO_DATA),
        "fields": len(config.BIO_DATA.get("_flat", {})),
        "prompt": config.BIO_DATA_PROMPT,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        loop="asyncio",
    )

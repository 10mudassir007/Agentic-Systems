from fastapi import APIRouter, HTTPException
from app.schemas import ChartPromptRequest
from app.agent.graph import run_agent, extract_chart_config

router = APIRouter(prefix="/chart", tags=["chart"])


@router.post("/prompt")
def chart_prompt(body: ChartPromptRequest):
    result = run_agent(body.prompt)
    messages = result.get("messages", [])
    config = extract_chart_config(messages)

    if "error" in config:
        raise HTTPException(status_code=422, detail=config)

    return config

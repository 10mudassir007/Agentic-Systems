import json
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, ToolMessage
from langchain.agents import create_agent
from app.config import GEMINI_API_KEY, GROQ_API_KEY
from app.db.connection import DBState
from app.agent.tools import run_sql, render_chart
from langchain.agents.middleware import ModelFallbackMiddleware

logger = logging.getLogger(__name__)

tools = [run_sql, render_chart]

fallback = ModelFallbackMiddleware(first_model=ChatGroq(model="openai/gpt-oss-120b", api_key=GROQ_API_KEY,temperature=0))

SYSTEM_PROMPT_TEMPLATE = """You are a data analyst assistant connected to a PostgreSQL database.

Database schema:
{schema}

You can write and execute SQL queries using the run_sql tool. Follow these rules:
- Only generate SELECT statements — never INSERT, UPDATE, DELETE, DROP, or any DDL.
- Use run_sql to explore data first. You may call it multiple times to refine.
- When you have the data you need, call render_chart with the chart type and field mapping.
- If run_sql returns an ERROR, read the error message carefully and try a corrected query.
- Choose appropriate chart types: bar, line, pie, scatter.
- The data_json parameter in render_chart must be the JSON string returned by run_sql.
- Set a descriptive title for the chart."""


def build_agent():
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        api_key=GEMINI_API_KEY,
    )
    schema = DBState.schema_text or "(no schema loaded)"
    system = SYSTEM_PROMPT_TEMPLATE.format(schema=schema)
    return create_agent(llm, tools, system_prompt=system, middleware=[fallback])


def run_agent(prompt: str):
    logger.info(f"Incoming prompt: {prompt}")
    agent = build_agent()
    result = agent.invoke({"messages": [("user", prompt)]})
    return result


def extract_chart_config(messages) -> dict:
    render_call = None
    last_run_sql_data = None
    print(render_call)
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "render_chart":
                    render_call = tc["args"]
                elif tc["name"] == "run_sql":
                    pass
        elif isinstance(msg, ToolMessage):
            if msg.name == "run_sql" and msg.content:
                try:
                    json.loads(msg.content)
                    last_run_sql_data = msg.content
                except (json.JSONDecodeError, TypeError):
                    pass

        if render_call is not None:
            break

    if render_call is None:
        return {"error": "agent did not produce a chart configuration"}

    raw_data = render_call.get("data_json") or last_run_sql_data or "[]"
    try:
        data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    except (json.JSONDecodeError, TypeError):
        data = []

    x_field = render_call.get("x_field", "")
    y_field = render_call.get("y_field", "")

    if data and isinstance(data, list) and len(data) > 0:
        keys = set(data[0].keys()) if isinstance(data[0], dict) else set()
        if x_field and x_field not in keys:
            return {"error": f"x_field '{x_field}' not found in data; available keys: {sorted(keys)}"}
        if y_field and y_field not in keys:
            return {"error": f"y_field '{y_field}' not found in data; available keys: {sorted(keys)}"}

    return {
        "chart_type": render_call.get("chart_type", "bar"),
        "x_field": x_field,
        "y_field": y_field,
        "title": render_call.get("title", ""),
        "data": data,
    }

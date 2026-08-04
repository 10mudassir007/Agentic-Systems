import json
import logging
import pandas as pd
from langchain_core.tools import tool
from app.db.connection import DBState
from app.agent.safety import is_safe_query

logger = logging.getLogger(__name__)


@tool
def run_sql(query: str) -> str:
    """Execute a SELECT SQL query against the connected database and return the results as JSON."""
    safe, result = is_safe_query(query)
    if not safe:
        logger.error(f"Unsafe query rejected: {result}")
        return f"ERROR: {result}"

    logger.info(f"Executing SQL: {result}")
    if DBState.engine is None:
        return "ERROR: no database connected"

    try:
        df = pd.read_sql(result, DBState.engine)
        return df.to_json(orient="records")
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        return f"ERROR executing query: {e}"


@tool
def render_chart(chart_type: str, data_json: str, x_field: str, y_field: str, title: str = "") -> str:
    """Declare the final chart configuration. chart_type: bar, line, pie, scatter. data_json is the JSON data string from run_sql."""
    payload = {
        "chart_type": chart_type,
        "data_json": data_json,
        "x_field": x_field,
        "y_field": y_field,
        "title": title,
    }
    return json.dumps(payload)

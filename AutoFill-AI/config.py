import os
import re
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

def load_bio_data() -> Dict[str, Any]:
    """Parse my_data.md into a structured bio data dictionary.

    Bio data is loaded once at startup and consumed only by the
    Data Mapper agent. It never enters the shared LangGraph state,
    so no other agent can accidentally access it.
    """
    md_path = Path(__file__).parent / "my_data.md"

    try:
        text = md_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"_raw": "No bio data file found.", "_flat": {}}

    bio: Dict[str, Any] = {}
    current_section = "general"

    for line in text.splitlines():
        if line.startswith("## "):
            label = line.strip("## ").strip().lower().replace(" ", "_").replace("/", "_")
            current_section = label
            bio.setdefault(current_section, {})
        elif line.startswith("- **"):
            m = re.match(r"- \*\*(.+?):?\*\*\s*(.*)", line)
            if m:
                key = m.group(1).strip().lower().replace(" ", "_").replace("/", "_")
                val = m.group(2).strip()
                target = bio[current_section] if isinstance(bio.get(current_section), dict) else bio
                target[key] = val

    flat: Dict[str, str] = {}
    for v in bio.values():
        if isinstance(v, dict):
            flat.update(v)
    bio["_flat"] = flat

    return bio


def format_bio_data_for_prompt(bio: Dict[str, Any]) -> str:
    """Render bio data as a readable string for use in LLM system prompts."""
    lines: list[str] = []
    for section, data in bio.items():
        if section == "_flat":
            continue
        if isinstance(data, dict) and data:
            lines.append(f"\n{section}")
            for k, v in data.items():
                lines.append(f"  {k}: {v}")
        elif isinstance(data, str):
            lines.append(f"{section}: {data}")
    return "\n".join(lines)


class Config:
    # --- Gemini ---
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    # --- Pipeline ---
    MAX_RETRIES_PER_FIELD: int = int(os.getenv("MAX_RETRIES_PER_FIELD", "3"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "5"))

    # --- Server ---
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    def __init__(self) -> None:
        self._bio_data: Dict[str, Any] = load_bio_data()
        self._bio_data_prompt: str = format_bio_data_for_prompt(self._bio_data)

    @property
    def BIO_DATA(self) -> Dict[str, Any]:
        return self._bio_data

    @property
    def BIO_DATA_PROMPT(self) -> str:
        return self._bio_data_prompt

    def update_bio_data(self, new_data: Dict[str, Any]) -> None:
        self._bio_data = new_data
        self._bio_data_prompt = format_bio_data_for_prompt(new_data)


config = Config()

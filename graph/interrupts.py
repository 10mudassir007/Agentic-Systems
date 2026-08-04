# backend/graph/interrupts.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class HumanInterrupt(BaseModel):
    """Model for LangGraph interrupt data."""
    run_id: str
    questions: List[Dict[str, Any]]  # List of question objects
    prompt: str = "Human input required"
    
    def dict(self):
        return {
            "run_id": self.run_id,
            "questions": self.questions,
            "prompt": self.prompt
        }

def human_interrupt(run_id: str, questions: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Helper function to create a human interrupt.
    Returns a dictionary of field_id -> human answer.
    
    Note: This is meant to be called from within a LangGraph node
    using the interrupt() function from langgraph.types.
    """
    return {
        "run_id": run_id,
        "questions": questions,
        "type": "human_input_request"
    }
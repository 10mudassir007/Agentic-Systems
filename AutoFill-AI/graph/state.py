from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict


class FieldStatus(str, Enum):
    PENDING = "pending"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    FILLED = "filled"
    FILL_ERROR = "fill_error"


class RunStatus(str, Enum):
    CREATED = "created"
    READING_FORM = "reading_form"
    MAPPING_FIELDS = "mapping_fields"
    WAITING_FOR_HUMAN = "waiting_for_human"
    FILLING_FORM = "filling_form"
    COMPLETED = "completed"
    FAILED = "failed"


class Field(TypedDict):
    field_id: str
    field_label: str
    field_type: str
    is_required: bool
    options: Optional[List[str]]
    selector: Optional[str]
    placeholder: Optional[str]
    validation_type: Optional[str]


class FieldResult(TypedDict):
    field_id: str
    status: FieldStatus
    value: Optional[str]
    confidence: Optional[float]
    error_message: Optional[str]
    human_question: Optional[str]
    timestamp: Optional[str]


class GraphState(TypedDict):
    """Shared graph state.

    Bio data is deliberately absent from this state object.
    Only the Data Mapper agent loads it (from config.BIO_DATA),
    keeping the blast radius contained.
    """

    run_id: str
    status: RunStatus
    form_url: str
    fields: List[Field]
    fields_loaded: bool
    resolved_fields: Dict[str, FieldResult]
    unresolved_fields: Dict[str, FieldResult]
    filler_results: Dict[str, FieldResult]
    filler_errors: List[str]
    awaiting_human: bool
    human_questions: List[str]
    human_answers: Dict[str, str]
    retry_count: int
    max_retries: int
    field_retry_counts: Dict[str, int]
    form_reader_attempts: int
    mapping_attempts: int
    submitted: bool
    completion_message: Optional[str]
    error_message: Optional[str]
    next_agent: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]


def make_initial_state(run_id: str, form_url: str, max_retries: int = 3) -> GraphState:
    return GraphState(
        run_id=run_id,
        status=RunStatus.CREATED,
        form_url=form_url,
        fields=[],
        fields_loaded=False,
        resolved_fields={},
        unresolved_fields={},
        filler_results={},
        filler_errors=[],
        awaiting_human=False,
        human_questions=[],
        human_answers={},
        retry_count=0,
        max_retries=max_retries,
        field_retry_counts={},
        form_reader_attempts=0,
        mapping_attempts=0,
        submitted=False,
        completion_message=None,
        error_message=None,
        next_agent=None,
        started_at=None,
        completed_at=None,
    )

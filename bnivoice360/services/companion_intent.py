"""Typed workflow arguments and schema validation; language interpretation belongs to the LLM."""

from dataclasses import dataclass

from bnivoice360.config import CATEGORY_COLUMNS, SENTIMENTS, SOURCES
from bnivoice360.services.agent.guardrails import GuardrailError


def object_schema(properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


INTENT_SCHEMA = object_schema(
    {
        "action": {
            "type": "string",
            "enum": [
                "search",
                "period",
                "select",
                "similar",
                "escalate",
                "details",
                "cancel",
                "aggregate",
                "help",
                "exists",
                "display",
                "dismiss",
                "revise",
            ],
        },
        "keyword": {"type": ["string", "null"]},
        "period": {
            "type": ["string", "null"],
            "enum": ["7d", "30d", "today", "this_week", "this_month", "all", "custom", None],
        },
        "start": {"type": ["string", "null"]},
        "end": {"type": ["string", "null"]},
        "sources": {"type": "array", "items": {"type": "string", "enum": list(SOURCES)}},
        "status": {
            "type": ["string", "null"],
            "enum": ["All", "Active", "Open", "Escalated", "Resolved", None],
        },
        "indices": {"type": "array", "items": {"type": "integer"}},
        "team": {"type": ["string", "null"], "enum": ["IT", "Operasional", "Bisnis", None]},
        "request": {"type": ["string", "null"]},
        "examination": {"type": ["string", "null"]},
        "cross_topic": {"type": "boolean"},
        "categories": {
            "type": ["array", "null"],
            "items": {"type": "string", "enum": list(CATEGORY_COLUMNS)},
        },
        "sentiments": {
            "type": ["array", "null"],
            "items": {"type": "string", "enum": list(SENTIMENTS)},
        },
        "urgent_only": {"type": ["boolean", "null"]},
        "refinement": {"type": "boolean"},
    }
)


def validate_schema(value, schema: dict):
    """Validate the small supported schema vocabulary again at the application boundary."""
    types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
    matches = {
        "null": value is None,
        "string": isinstance(value, str),
        "integer": type(value) is int,
        "boolean": type(value) is bool,
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }
    if not any(matches[kind] for kind in types) or (
        "enum" in schema and value not in schema["enum"]
    ):
        raise GuardrailError("invalid_structured_output")
    if isinstance(value, dict):
        if not set(value) <= set(schema["properties"]) or not set(schema["required"]) <= set(value):
            raise GuardrailError("invalid_structured_output")
        for key, item in value.items():
            validate_schema(item, schema["properties"][key])
    elif isinstance(value, list):
        if len(value) > 51:
            raise GuardrailError("structured_list_budget")
        for item in value:
            validate_schema(item, schema["items"])


@dataclass(frozen=True)
class Intent:
    action: str
    keyword: str | None = None
    period: str | None = None
    start: str | None = None
    end: str | None = None
    sources: tuple[str, ...] = ()
    status: str | None = None
    indices: tuple[int, ...] = ()
    team: str | None = None
    request: str | None = None
    examination: str | None = None
    cross_topic: bool = False
    categories: tuple[str, ...] | None = None
    sentiments: tuple[str, ...] | None = None
    urgent_only: bool | None = None
    refinement: bool = False

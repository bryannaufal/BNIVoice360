"""Read only source-provided masked complaint text for external model payloads."""

import pandas as pd

from bnivoice360.services.agent.guardrails import TOXIC, GuardrailError, screen_input


def masked_text(row) -> str:
    """Fail closed for missing masks; never substitute the original content."""
    value = row.get("masked_content")
    if value is None or pd.isna(value) or not str(value).strip():
        raise GuardrailError("masked_content_unavailable")
    text = screen_input(str(value))
    if TOXIC.search(text):
        raise GuardrailError("unsafe_source_text")
    return text

"""Resolve supported Groq secret formats without copying secrets into session state."""

import os
from collections.abc import Mapping

import streamlit as st


def resolve_groq_key(secrets: Mapping) -> str:
    for name in ("GROQ_API_KEY", "GROQ", "groq"):
        value = secrets.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Mapping):
            nested = value.get("api_key") or value.get("GROQ_API_KEY")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return ""


def groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        return resolve_groq_key(st.secrets)
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return ""

"""Local layered screening and strict output validation, without PII masking."""

import json
import re
import unicodedata


class GuardrailError(ValueError):
    """Rejected content; only a safe reason code is logged."""


INJECTION = re.compile(
    r"ignore\s+(?:all\s+)?(?:previous|prior|system)|abaikan\s+(?:semua\s+)?(?:instruksi|perintah)"
    r"|system\s*prompt|developer\s*mode|reveal\s+(?:your\s+)?(?:secret|key|prompt)"
    r"|bypass\s+(?:guardrails?|safety)|<\|(?:system|im_start)"
    r"|(?:hapus|delete|drop)\s+(?:semua\s+|all\s+)?(?:data|table|database)"
    r"|(?:send|kirim|upload)\b.{0,40}\b(?:api\s*key|secret|database)",
    re.I,
)
TOXIC = re.compile(
    r"\b(?:bunuh\s+(?:nasabah|mereka)|kill\s+(?:customers|them)|"
    r"(?:dasar\s+)?(?:goblok|bajingan|kontol)|fuck(?:ing)?|nigger)\b",
    re.I,
)
LEAK = re.compile(
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\d)\d(?:[ -]?\d){15}(?!\d)"
    r"|(?<!\w)(?:\+62|62|0)[ -]?8(?:[ -]?\d){8,11}(?!\d)"
    r"|\b(?:gsk_|sk-)[A-Za-z0-9_-]{8,}",
    re.I,
)
OUT_OF_SCOPE = re.compile(
    r"\b(?:resep|recipe|weather|cuaca|horoscope|horoskop|crypto|politik|politics)\b", re.I
)
SCOPE = re.compile(
    r"keluhan|komplain|isu|issue|complaint|eskalasi|escalat|tindak\s+lanjut|investigas|ringkas|summar",
    re.I,
)


def normalized(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKC", text) if unicodedata.category(c) != "Cf"
    )


def screen_input(text: str, *, instruction: bool = False) -> str:
    if (
        not isinstance(text, str)
        or not text.strip()
        or len(text) > (1000 if instruction else 12000)
    ):
        raise GuardrailError("input_size")
    clean = normalized(text)
    if INJECTION.search(clean):
        raise GuardrailError("prompt_injection")
    if instruction and (OUT_OF_SCOPE.search(clean) or not SCOPE.search(clean)):
        raise GuardrailError("outside_complaint_scope")
    if LEAK.search(clean):
        raise GuardrailError("structured_pii_or_secret_input")
    if re.search(r"https?://|<script|javascript:", clean, re.I):
        raise GuardrailError("unsafe_link_or_markup_input")
    return clean


def validate_output(raw: str, evidence: dict[str, str]) -> dict:
    if not isinstance(raw, str) or len(raw) > 14000:
        raise GuardrailError("output_size")
    clean = normalized(raw)
    if LEAK.search(clean):
        raise GuardrailError("structured_pii_or_secret_output")
    if TOXIC.search(clean) or INJECTION.search(clean):
        raise GuardrailError("unsafe_output")
    if re.search(r"https?://|<[^>]+>|javascript:", clean, re.I):
        raise GuardrailError("unsafe_output_markup")
    try:
        result = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise GuardrailError("invalid_json") from exc
    if not isinstance(result, dict) or set(result) != {"findings", "hypotheses", "actions"}:
        raise GuardrailError("invalid_schema")
    decoded = normalized(json.dumps(result, ensure_ascii=False))
    if LEAK.search(decoded):
        raise GuardrailError("structured_pii_or_secret_output")
    if TOXIC.search(decoded) or INJECTION.search(decoded):
        raise GuardrailError("unsafe_output")
    if re.search(r"https?://|<[^>]+>|javascript:", decoded, re.I):
        raise GuardrailError("unsafe_output_markup")
    for kind in ("findings", "hypotheses", "actions"):
        items = result[kind]
        if not isinstance(items, list) or len(items) > 5 or (kind == "actions" and not items):
            raise GuardrailError("invalid_schema")
        for item in items:
            required = {"text", "evidence"} | ({"team"} if kind == "actions" else set())
            if not isinstance(item, dict) or set(item) != required:
                raise GuardrailError("invalid_schema")
            if not isinstance(item["text"], str) or not 1 <= len(item["text"]) <= 600:
                raise GuardrailError("invalid_schema")
            # Counts come from application code, never generated claims.
            if re.search(r"\d", item["text"]):
                raise GuardrailError("unverified_number")
            if kind == "actions" and item["team"] not in {"IT", "Operasional", "Bisnis"}:
                raise GuardrailError("invalid_team")
            references = item["evidence"]
            if not isinstance(references, list) or not 1 <= len(references) <= len(evidence):
                raise GuardrailError("missing_evidence")
            for reference in references:
                if not isinstance(reference, dict) or set(reference) != {"ticket", "quote"}:
                    raise GuardrailError("invalid_reference")
                alias, quote = reference["ticket"], reference["quote"]
                if not isinstance(alias, str) or alias not in evidence:
                    raise GuardrailError("invented_reference")
                if (
                    not isinstance(quote, str)
                    or len(quote.strip()) < 8
                    or quote not in evidence[alias]
                ):
                    raise GuardrailError("ungrounded_quote")
    return result

"""Provider metrics without prompts, responses, credentials or duplicate token totals."""

import time
import uuid


class AuditedModelClient:
    def __init__(self, client, audit, session_id: str, model: str):
        self.client, self.audit = client, audit
        self.session_id, self.model = session_id, model

    def _call(self, operation, method, *args, **kwargs):
        call_id = str(uuid.uuid4())
        started = time.monotonic()
        self.audit.log(
            self.session_id,
            "model_call_started",
            call_id=call_id,
            operation=operation,
            model=self.model,
        )
        try:
            result, usage = method(*args, **kwargs)
        except Exception as exc:
            self.audit.log(
                self.session_id,
                "model_call_finished",
                call_id=call_id,
                operation=operation,
                model=self.model,
                status="failed",
                error_code=type(exc).__name__,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            raise
        tokens = usage.get("tokens")
        self.audit.log(
            self.session_id,
            "model_call_finished",
            call_id=call_id,
            operation=operation,
            model=self.model,
            status="ok",
            tokens=tokens if type(tokens) is int and tokens >= 0 else None,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        return result, usage

    def structured(self, system, data, schema, name, **kwargs):
        return self._call(name, self.client.structured, system, data, schema, name, **kwargs)

    def choose_tools(self, messages, definitions, **kwargs):
        return self._call("tool_planner", self.client.choose_tools, messages, definitions, **kwargs)

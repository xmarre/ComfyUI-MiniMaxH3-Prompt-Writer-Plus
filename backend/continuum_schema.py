from __future__ import annotations

from typing import Any


CONTINUUM_PLAN_SCHEMA_VERSION = 2


def continuum_plan_json_schema(settings: dict[str, Any]) -> dict[str, Any]:
    """Return the portable structural contract used for Continuum planner output."""
    chunks = settings.get("chunks")
    if not isinstance(chunks, int) or isinstance(chunks, bool) or chunks < 1:
        raise ValueError("Continuum planner schema requires a positive integer chunk count.")

    subject_anchor = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {
                "type": "string",
                "pattern": r"^<Subject [1-9][0-9]*>$",
            },
            "meaning": {"type": "string", "minLength": 1},
        },
        "required": ["id", "meaning"],
    }
    chunk = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "continuity": {
                "type": "string",
                "enum": ["initial", "continuous", "intentional_break"],
            },
            "transition": {"type": "string"},
            "start_state": {"type": "string", "minLength": 1},
            "action": {"type": "string", "minLength": 1},
            "end_state": {"type": "string", "minLength": 1},
        },
        "required": ["continuity", "transition", "start_state", "action", "end_state"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": CONTINUUM_PLAN_SCHEMA_VERSION},
            "global": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "sequence_preamble": {"type": "string", "minLength": 1},
                    "continuity_anchors": {"type": "string"},
                    "persistent_constraints": {"type": "string"},
                    "subject_anchors": {
                        "type": "array",
                        "items": subject_anchor,
                    },
                },
                "required": [
                    "sequence_preamble",
                    "continuity_anchors",
                    "persistent_constraints",
                    "subject_anchors",
                ],
            },
            "chunks": {
                "type": "array",
                "items": chunk,
                "minItems": chunks,
                "maxItems": chunks,
            },
        },
        "required": ["schema_version", "global", "chunks"],
    }


def planner_response_metadata(result: dict[str, Any]) -> dict[str, Any]:
    """Return non-content diagnostics for a planner model response."""
    text = str(result.get("prompt") or "")
    stripped = text.lstrip()
    lowered = stripped.casefold()
    finish_reason = result.get("primary_finish_reason")
    return {
        "chars": len(text),
        "finish_reason": str(finish_reason) if finish_reason is not None else None,
        "starts_with_object": stripped.startswith("{"),
        "starts_with_array": stripped.startswith("["),
        "starts_with_fence": stripped.startswith("```"),
        "starts_with_think": lowered.startswith("<think"),
        "contains_object_open": "{" in text,
        "contains_object_close": "}" in text,
    }

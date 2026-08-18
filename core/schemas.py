"""JSON schemas for constraining the evaluated model's reply.

Only used when ``--structured_output true``. Two task families deliberately
return ``None`` and stay free-form no matter what:

* instruction following (IFBench) — the reply's *form* is what is being graded,
  so imposing a shape would grade the schema rather than the model;
* the safety probes (Qorgau) — the question is the probe, and a JSON contract
  changes what the model is being asked to do.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .parsing import LETTERS


def schema(name: str, properties: Dict[str, Any],
           required: List[str]) -> Dict[str, Any]:
    """An OpenAI/vLLM strict json_schema response_format."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def mcq(n_options: int) -> Dict[str, Any]:
    """One letter, drawn only from the options this item actually has."""
    letters = LETTERS[:max(1, min(n_options, 26))]
    return schema("mcq_answer",
                  {"answer": {"type": "string", "enum": letters}}, ["answer"])


def number() -> Dict[str, Any]:
    return schema("numeric_answer",
                  {"answer": {"type": "number"}}, ["answer"])


def text(name: str = "answer", key: str = "answer") -> Dict[str, Any]:
    return schema(name, {key: {"type": "string"}}, [key])


def reasoned_text(key: str = "answer") -> Dict[str, Any]:
    """A short rationale then the answer.

    Constrained decoding cannot emit a <think> block, so for tasks that benefit
    from working through the problem this keeps a place for it inside the JSON.
    """
    return schema("reasoned_answer",
                  {"reasoning": {"type": "string"}, key: {"type": "string"}},
                  ["reasoning", key])

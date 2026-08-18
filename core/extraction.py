"""Splitting reasoning from answers, and pulling the answer out of the answer."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, List, Optional

from .parsing import LETTERS


@dataclass
class Completion:
    """One model turn, with the reasoning trace separated from the answer."""

    reasoning: str = ""
    answer: str = ""
    # True when a thinking block was opened but never closed, i.e. generation
    # hit the token limit mid-thought and there is no answer to score.
    truncated_thinking: bool = False
    finish_reason: str = ""

    @property
    def for_extraction(self) -> str:
        """Text a short answer may be pulled from.

        Normally the answer. When a reasoning parser puts everything in the
        reasoning channel and leaves ``content`` empty, the answer is in there
        too — recovering it beats losing the item. Only for tasks that extract
        a letter or a number; free-form tasks (ASR, translation, captioning,
        safety) must never read the reasoning as if it were the reply.
        """
        return self.answer if self.answer.strip() else self.reasoning


# Servers expose the separated trace under either name, and the OpenAI SDK may
# surface a server-added field only inside `model_extra` rather than as a real
# attribute — so both names and both locations have to be checked.
REASONING_FIELDS = ("reasoning_content", "reasoning")


def reasoning_from_message(message: Any) -> str:
    """The reasoning text from a chat message, wherever the SDK put it."""
    if message is None:
        return ""
    extra = getattr(message, "model_extra", None) or {}
    if not isinstance(extra, dict):
        extra = {}
    for name in REASONING_FIELDS:
        value = getattr(message, name, None)
        if value is None:
            value = extra.get(name)
        if value and str(value).strip():
            return str(value).strip()
    return ""


# Channel / thinking markup emitted by various families, e.g.
#   <think>...</think>, <|think|>, <|channel|>analysis<|message|>, <channel|>
_MARKUP = re.compile(r"<\|[^>]*?\|?>|<[^>]*?\|>|</?think>")


def strip_markup(text: str) -> str:
    """Remove leftover channel/thinking tokens from an answer."""
    return _MARKUP.sub(" ", text or "").strip()


def split_reasoning(content: str, reasoning_content: Optional[str],
                    think_start: str, think_end: str) -> Completion:
    """Normalise the ways a served model can expose its reasoning.

    With a vLLM ``--reasoning-parser`` the trace arrives out-of-band and
    ``content`` is already clean. Without one, it is inline and delimited by
    the thinking tokens — or wrapped in channel markup, which is stripped.
    """
    content = content or ""

    if reasoning_content:
        answer = strip_markup(content)
        # A parser that routed everything into the reasoning channel leaves
        # content empty. That is not a truncated thought: the reply is there,
        # just in the other field, and `for_extraction` reaches it.
        return Completion(reasoning=reasoning_content.strip(), answer=answer,
                          truncated_thinking=False)

    if think_end and think_end in content:
        head, _, tail = content.rpartition(think_end)
        if think_start and think_start in head:
            head = head.split(think_start, 1)[1]
        return Completion(reasoning=strip_markup(head),
                          answer=strip_markup(tail))

    if think_start and think_start in content:
        # Opened but never closed -> everything is reasoning. The answer may
        # still be recoverable from it, so this is flagged rather than dropped.
        return Completion(reasoning=strip_markup(content.split(think_start, 1)[1]),
                          answer="", truncated_thinking=True)

    return Completion(reasoning="", answer=strip_markup(content))


def find_json_objects(text: str) -> List[str]:
    """Return every balanced ``{...}`` span in ``text``, outermost first."""
    spans: List[str] = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth, in_string, escaped, start = 0, False, False, i
        for j in range(i, len(text)):
            ch = text[j]
            if escaped:
                escaped = False
                continue
            if ch == "\\" and in_string:
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    spans.append(text[start:j + 1])
                    i = j + 1
                    break
        else:
            break
    return spans


def _json_answers(text: str) -> List[str]:
    """Values of the ``answer`` key from every JSON object in ``text``."""
    values: List[str] = []
    for span in find_json_objects(text):
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("answer") is not None:
            values.append(str(obj["answer"]))
    return values


def extract_mcq(answer: str, n_options: int = 26) -> Optional[str]:
    """Pull the chosen option letter out of a model answer."""
    if not answer:
        return None
    max_letter = LETTERS[max(0, min(n_options, 26) - 1)]
    pattern = f"[A-{max_letter}]"

    for value in reversed(_json_answers(answer)):
        match = re.search(f"({pattern})", value.upper())
        if match:
            return match.group(1)

    # Letter matching is case-insensitive: SAKURA (and several audio/vision
    # benchmarks) explicitly instruct the model to reply with a *lowercase*
    # letter, so a case-sensitive [A-D] would score every such answer as a
    # failed extraction.
    labelled = re.findall(rf"(?:answer|жауап|ответ)\s*[:\-\s]*\**({pattern})\b",
                          answer, flags=re.IGNORECASE)
    if labelled:
        return labelled[-1].upper()

    boxed = re.findall(r"\\boxed\{\s*([A-Za-z])\s*\}", answer)
    if boxed and boxed[-1].upper() <= max_letter:
        return boxed[-1].upper()

    tail = re.search(rf"\b({pattern})\)?\s*\.?\s*$", answer.strip(),
                     flags=re.IGNORECASE)
    if tail:
        return tail.group(1).upper()

    return None


def extract_numeric(answer: str) -> Optional[float]:
    """Pull a single number out of a model answer."""
    if not answer:
        return None

    def _first_number(text: str) -> Optional[float]:
        cleaned = text.replace(",", "").replace("$", "").replace("%", "")
        numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        return float(numbers[-1]) if numbers else None

    for value in reversed(_json_answers(answer)):
        number = _first_number(value)
        if number is not None:
            return number

    boxed = extract_boxed(answer)
    if boxed:
        number = _first_number(boxed)
        if number is not None:
            return number

    hashed = re.search(r"####\s*(-?[\d,\.]+)", answer)
    if hashed:
        return _first_number(hashed.group(1))

    return _first_number(answer)


def extract_boxed(answer: str) -> Optional[str]:
    """Return the contents of the last ``\\boxed{...}``, brace-balanced."""
    if not answer:
        return None
    marker = "\\boxed"
    index = answer.rfind(marker)
    if index == -1:
        return None
    rest = answer[index + len(marker):].lstrip()
    if not rest.startswith("{"):
        # \boxed 42 — rare, but cheap to support.
        token = rest.split()[0] if rest.split() else ""
        return token or None
    depth = 0
    for pos, ch in enumerate(rest):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return rest[1:pos].strip()
    return rest[1:].strip() or None


def extract_flexible(answer: str) -> Optional[str]:
    """For answers that may be a letter *or* a value.

    Prefers a JSON ``answer`` field, then an "Answer: ..." line, and finally
    falls back to the last non-empty line — which is where a model that ignored
    the format instruction usually leaves its conclusion.
    """
    if not answer:
        return None

    values = _json_answers(answer)
    if values:
        return values[-1].strip()

    labelled = re.findall(
        r"(?i:answer|жауап|ответ)\s*[:\-]\s*(.+?)(?:\n|$)", answer
    )
    if labelled:
        return labelled[-1].strip().strip('"').strip("'")

    boxed = extract_boxed(answer)
    if boxed:
        return boxed

    lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
    return lines[-1] if lines else None


def extract_freeform(answer: str) -> Optional[str]:
    """Prefer a JSON ``answer`` field, else return the answer text as-is."""
    if not answer:
        return None
    values = _json_answers(answer)
    if values:
        return values[-1].strip()
    return answer.strip() or None

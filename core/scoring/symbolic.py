"""Symbolic answer checking for PolyMath — sympy instead of an LLM judge.

The verdict is deliberately three-valued.  ``correct=None`` means "this pair
could not be decided symbolically"; the runner reports those items as
*unscorable* and excludes them from the denominator rather than silently
counting them wrong.
"""

from __future__ import annotations

import re
import signal
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

# math-verify (HuggingFace) is the strongest LaTeX-aware checker available; it
# is used when installed and this module's own logic is the fallback.
try:  # pragma: no cover - depends on the environment
    from math_verify import parse as _mv_parse, verify as _mv_verify

    _HAS_MATH_VERIFY = True
except Exception:  # noqa: BLE001
    _HAS_MATH_VERIFY = False

TOLERANCE = 1e-6


@dataclass
class Verdict:
    correct: Optional[bool]     # None -> unscorable
    method: str                 # which check decided it
    prediction: str = ""
    reference: str = ""


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_STRIP_WRAPPERS = [
    (re.compile(r"\\(?:text|mathrm|mathbf|textbf|mbox|operatorname)\s*\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\\left\s*"), ""),
    (re.compile(r"\\right\s*"), ""),
    (re.compile(r"\\[!,;:> ]"), ""),
    (re.compile(r"\\%"), ""),
    (re.compile(r"\\\$"), ""),
    (re.compile(r"\^\s*\{?\\circ\}?"), ""),
    (re.compile(r"\\(?:d|t)frac"), r"\\frac"),
    (re.compile(r"\s+"), " "),
]

_TRAILING = " \t\n.,;:$\u00a0"

# How deep element-wise tuple/set comparison may recurse.
_MAX_TUPLE_DEPTH = 8


def normalize(text: Any) -> str:
    """Reduce a LaTeX-ish answer to a canonical comparable string."""
    value = str(text or "").strip()
    value = value.replace("$", " ").replace("\\\\", " ")
    # A stray \boxed{...} survives when the caller passed raw model text.
    boxed = re.fullmatch(r"\s*\\boxed\s*\{(.*)\}\s*", value, flags=re.DOTALL)
    if boxed:
        value = boxed.group(1)
    for pattern, replacement in _STRIP_WRAPPERS:
        value = pattern.sub(replacement, value)
    value = value.strip(_TRAILING).strip()
    # "x = 5" / "answer: 5" -> "5"
    value = re.sub(r"^[A-Za-z]\s*=\s*", "", value)
    return value.strip()


def _to_number(text: str) -> Optional[float]:
    """Parse plain numbers, percentages and simple LaTeX fractions."""
    value = text.replace(" ", "").replace(",", "")
    if not value:
        return None

    fraction = re.fullmatch(r"-?\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", value)
    if fraction:
        try:
            result = float(fraction.group(1)) / float(fraction.group(2))
            return -result if value.startswith("-") else result
        except (ValueError, ZeroDivisionError):
            return None

    simple = re.fullmatch(r"(-?\d+)/(-?\d+)", value)
    if simple:
        try:
            return float(simple.group(1)) / float(simple.group(2))
        except (ValueError, ZeroDivisionError):
            return None

    value = value.rstrip("%")
    try:
        return float(value)
    except ValueError:
        return None


def _split_tuple(text: str) -> Optional[List[str]]:
    """Split ``(a, b)``/``{a, b}``/``a, b`` into parts, if it is a collection.

    Returns None unless the split produced several parts: a string whose only
    commas are nested (``f(x, y)``) yields one part identical to the input,
    and recursing on that never terminates.
    """
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] in "({[" and stripped[-1] in ")}]":
        stripped = stripped[1:-1]
    if "," not in stripped:
        return None
    parts, depth, current = [], 0, []
    for ch in stripped:
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    parts.append("".join(current).strip())
    parts = [p for p in parts if p]
    # Fewer than two parts means no progress was made; see the docstring.
    return parts if len(parts) >= 2 else None


# ---------------------------------------------------------------------------
# LaTeX -> sympy
# ---------------------------------------------------------------------------

_LATEX_TO_PY = [
    (re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}"), r"((\1)/(\2))"),
    (re.compile(r"\\sqrt\[(\d+)\]\{([^{}]+)\}"), r"((\2)**(1/(\1)))"),
    (re.compile(r"\\sqrt\{([^{}]+)\}"), r"sqrt(\1)"),
    (re.compile(r"\\(?:cdot|times)"), "*"),
    (re.compile(r"\\div"), "/"),
    (re.compile(r"\\pi\b"), "pi"),
    (re.compile(r"\\infty"), "oo"),
    (re.compile(r"\\(?:ln|log|sin|cos|tan|exp)\b"), lambda m: m.group(0)[1:]),
    (re.compile(r"\^"), "**"),
    (re.compile(r"\{"), "("),
    (re.compile(r"\}"), ")"),
]


def _latex_to_sympy_source(text: str) -> str:
    value = text
    # Nested fractions need a few passes.
    for _ in range(4):
        for pattern, replacement in _LATEX_TO_PY:
            value = pattern.sub(replacement, value)
    return value.strip()


def _run_with_timeout(func, seconds: float):
    """Best-effort timeout: SIGALRM on the main thread, unguarded elsewhere."""
    if threading.current_thread() is not threading.main_thread():
        return func()

    def _raise(_signum, _frame):
        raise TimeoutError("symbolic check timed out")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return func()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _as_latex(text: str) -> str:
    """math-verify extracts from LaTeX, so make sure there is a math span."""
    stripped = str(text).strip()
    if stripped.startswith("$") and stripped.endswith("$") and len(stripped) > 1:
        return stripped
    return f"${stripped}$"


def _math_verify(prediction: str, reference: str) -> Optional[bool]:
    """Decide with math-verify, or return None if either side will not parse."""
    if not _HAS_MATH_VERIFY:
        return None
    try:
        gold = _mv_parse(_as_latex(reference))
        pred = _mv_parse(_as_latex(prediction))
    except Exception:  # noqa: BLE001
        return None
    if not gold or not pred:
        return None
    try:
        return bool(_mv_verify(gold, pred))
    except Exception:  # noqa: BLE001
        return None


def _sympy_equal(left: str, right: str) -> Optional[bool]:
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except Exception:  # noqa: BLE001
        return None

    transformations = standard_transformations + (
        implicit_multiplication_application,
    )

    def _parse(source: str):
        return parse_expr(_latex_to_sympy_source(source),
                          transformations=transformations, evaluate=True)

    def _compare():
        a, b = _parse(left), _parse(right)
        difference = sympy.simplify(a - b)
        if difference == 0:
            return True
        try:
            return bool(abs(complex(difference.evalf())) < TOLERANCE)
        except (TypeError, ValueError):
            return False

    try:
        return _run_with_timeout(_compare, 5.0)
    except Exception:  # noqa: BLE001 - unparseable is "undecided", not "wrong"
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def verify(prediction: Any, reference: Any, _depth: int = 0) -> Verdict:
    """Decide whether ``prediction`` matches ``reference``, or admit defeat.

    ``_depth`` bounds the element-wise recursion in step 4; beyond it the
    split is not converging and undecided is the safe answer.
    """
    if _depth > _MAX_TUPLE_DEPTH:
        return Verdict(None, "too_deep", str(prediction), str(reference))

    pred = normalize(prediction)
    ref = normalize(reference)

    if not pred:
        return Verdict(False, "empty", pred, ref)
    if not ref:
        return Verdict(None, "no_reference", pred, ref)

    # 1. Exact string match after normalisation.
    if pred == ref or pred.lower() == ref.lower():
        return Verdict(True, "exact", pred, ref)

    # 2. Numeric comparison.
    pred_number, ref_number = _to_number(pred), _to_number(ref)
    if pred_number is not None and ref_number is not None:
        scale = max(1.0, abs(ref_number))
        return Verdict(abs(pred_number - ref_number) <= TOLERANCE * scale,
                       "numeric", pred, ref)

    # 3. math-verify: a real LaTeX parser, so it handles \lfloor, \binom,
    #    matrices, sets and units that the heuristics below cannot.
    decided = _math_verify(str(prediction), str(reference))
    if decided is not None:
        return Verdict(decided, "math_verify", pred, ref)

    # 4. Element-wise comparison for tuples / sets / coordinate answers.
    pred_parts, ref_parts = _split_tuple(pred), _split_tuple(ref)
    if pred_parts and ref_parts:
        if len(pred_parts) != len(ref_parts):
            return Verdict(False, "tuple_length", pred, ref)
        verdicts = [verify(p, r, _depth + 1)
                    for p, r in zip(pred_parts, ref_parts)]
        if any(v.correct is None for v in verdicts):
            return Verdict(None, "tuple_undecided", pred, ref)
        return Verdict(all(v.correct for v in verdicts), "tuple", pred, ref)

    # 5. Symbolic equivalence via sympy.
    symbolic = _sympy_equal(pred, ref)
    if symbolic is not None:
        return Verdict(symbolic, "sympy", pred, ref)

    return Verdict(None, "unparseable", pred, ref)

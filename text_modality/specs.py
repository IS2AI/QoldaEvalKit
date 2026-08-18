"""Declarative benchmark definitions.

Language availability follows what actually exists on the Hub:

  kk + ru + en : mmlu_pro, gpqa, arc, gsm8k
  kk + en      : mmlu, mmlu_redux, polymath
  kk + ru      : include
  kk only      : kazculture, kazmmlu, kazbench_kk, belebele, piqa, kkcopa,
                 nis_math, kkwikispell, kazqad, ragbench, ifbench

English data comes from the upstream originals; where a stable item id exists
the English set is intersected with the Kazakh one (``align=True``) so that all
languages score an identical item set.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from core.parsing import (
    LETTERS,
    answer_key_to_index,
    gsm8k_gold,
    join_documents,
    parse_dict,
    parse_string_list,
    stable_shuffle,
)
from core.registry import BenchmarkSpec, Registry, Sample, Source, TaskType

REGISTRY = Registry()
register = REGISTRY.register

Row = Dict[str, Any]


# ---------------------------------------------------------------------------
# Adapters — one raw row -> one normalised Sample (or None to drop the row)
# ---------------------------------------------------------------------------

def _mcq(uid: str, question: str, options, gold_index: Optional[int],
         **kwargs) -> Optional[Sample]:
    opts = parse_string_list(options)
    if not opts or gold_index is None or not (0 <= gold_index < len(opts)):
        return None
    return Sample(uid=uid, question=str(question), options=opts,
                  reference=LETTERS[gold_index], **kwargs)


# --- MMLU (issai/MMLU_Kazakh and cais/mmlu share a schema) -----------------
def adapt_mmlu(row: Row, idx: int) -> Optional[Sample]:
    subject = str(row.get("subject", "unknown"))
    try:
        gold = int(row["answer"])
    except (KeyError, TypeError, ValueError):
        return None
    return _mcq(subject, row.get("question", ""), row.get("choices"), gold,
                group=subject)


# --- MMLU-Pro --------------------------------------------------------------
def adapt_mmlu_pro(row: Row, idx: int) -> Optional[Sample]:
    try:
        uid = str(row["question_id"])
        gold = int(row["answer_index"])
    except (KeyError, TypeError, ValueError):
        return None
    category = str(row.get("category", "unknown"))
    return _mcq(uid, row.get("question", ""), row.get("options"), gold,
                group=category)


# --- GPQA ------------------------------------------------------------------
def adapt_gpqa_translated(row: Row, idx: int) -> Optional[Sample]:
    """issai/GPQA_*: pre-shuffled choices with a 1-based ``answer``."""
    try:
        gold = int(row["answer"]) - 1
    except (KeyError, TypeError, ValueError):
        return None
    return _mcq(str(idx), row.get("question", ""), row.get("choices"), gold,
                group=str(row.get("subdomain", "unknown")))


def adapt_gpqa_original(row: Row, idx: int) -> Optional[Sample]:
    """Idavidrein/gpqa: one correct answer plus three distractors."""
    correct = row.get("Correct Answer")
    distractors = [row.get(f"Incorrect Answer {i}") for i in (1, 2, 3)]
    if correct is None or any(d is None for d in distractors):
        return None
    options = [str(correct)] + [str(d) for d in distractors]
    uid = str(idx)
    shuffled = stable_shuffle(options, uid)
    return _mcq(uid, row.get("Question", ""), shuffled,
                shuffled.index(str(correct)),
                group=str(row.get("Subdomain", "unknown")))


# --- ARC -------------------------------------------------------------------
def adapt_arc(row: Row, idx: int) -> Optional[Sample]:
    """Handles both issai (1-based int key) and allenai (letter key)."""
    choices = parse_dict(row.get("choices"))
    if not choices:
        return None
    options = [str(t) for t in choices.get("text", [])]
    if not options:
        return None
    # issai numbers options from 1; allenai uses letters (or 1-based ints).
    gold = answer_key_to_index(row.get("answerKey"), len(options), base=1)
    if gold is None:
        return None
    # allenai/ai2_arc has no `source` column; the loader fills the group in
    # from the config name and Source.group_map normalises it.
    source = row.get("source")
    return _mcq(str(row.get("id", idx)), row.get("question", ""), options, gold,
                group=str(source).lower() if source else None)


# --- GSM8K -----------------------------------------------------------------
def adapt_gsm8k(row: Row, idx: int) -> Optional[Sample]:
    gold = gsm8k_gold(row.get("answer_text", row.get("answer")))
    if gold is None:
        return None
    return Sample(uid=str(idx), question=str(row.get("question", "")),
                  reference=gold)


# --- MMLU-Redux 2.0 --------------------------------------------------------
def adapt_mmlu_redux_kk(row: Row, idx: int) -> Optional[Sample]:
    """issai ids look like ``abstract_algebra___0``; normalise to the
    ``subject#index`` form the English side builds with ``uid_counter``."""
    raw_id = str(row.get("id", ""))
    subject = str(row.get("subject", "unknown"))
    if "___" in raw_id:
        subject, _, position = raw_id.partition("___")
        uid = f"{subject}#{position}"
    else:
        uid = f"{subject}#{idx}"
    try:
        gold = int(row["answer"])
    except (KeyError, TypeError, ValueError):
        return None
    return _mcq(uid, row.get("question_kk", ""), row.get("choices_kk"), gold,
                group=subject)


def adapt_mmlu_redux_en(row: Row, idx: int) -> Optional[Sample]:
    try:
        gold = int(row["answer"])
    except (KeyError, TypeError, ValueError):
        return None
    # uid base only; the loader appends the per-subject counter.
    return _mcq("", row.get("question", ""), row.get("choices"), gold)


# --- PolyMath --------------------------------------------------------------
_POLYMATH_ID = re.compile(r"^(top|high|medium|low)-[a-z]{2}-(\d+)$")


def _polymath_uid(raw_id: Any, idx: int) -> str:
    match = _POLYMATH_ID.match(str(raw_id))
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return str(raw_id or idx)


def adapt_polymath_kk(row: Row, idx: int) -> Optional[Sample]:
    question = row.get("question_kk")
    if not question:
        return None
    uid = _polymath_uid(row.get("id"), idx)
    return Sample(uid=uid, question=str(question),
                  reference=str(row.get("answer", "")).strip(),
                  group=uid.split("-")[0])


def adapt_polymath_en(row: Row, idx: int) -> Optional[Sample]:
    question = row.get("question")
    if not question:
        return None
    uid = _polymath_uid(row.get("id"), idx)
    return Sample(uid=uid, question=str(question),
                  reference=str(row.get("answer", "")).strip(),
                  group=uid.split("-")[0])


# --- Kazakh-only MCQ sets --------------------------------------------------
def adapt_kazculture(row: Row, idx: int) -> Optional[Sample]:
    gold = answer_key_to_index(row.get("answer_label"), 4)   # letters A-D
    if gold is None:
        return None
    return _mcq(str(idx), row.get("question", ""),
                [row.get("a"), row.get("b"), row.get("c"), row.get("d")], gold)


def adapt_kazmmlu(row: Row, idx: int) -> Optional[Sample]:
    options = [row.get("Option A"), row.get("Option B"),
               row.get("Option C"), row.get("Option D")]
    if row.get("Option E"):
        options.append(row.get("Option E"))
    options = [o for o in options if o is not None]
    gold = answer_key_to_index(row.get("Answer Key"), len(options))   # letters
    if gold is None:
        return None
    return _mcq(str(idx), row.get("Question", ""), options, gold)


def adapt_kazbench_kk(row: Row, idx: int) -> Optional[Sample]:
    gold = answer_key_to_index(row.get("answer"), 4)   # letters A-D
    if gold is None:
        return None
    return _mcq(str(idx), row.get("question", ""),
                [row.get("A"), row.get("B"), row.get("C"), row.get("D")], gold,
                group=str(row.get("category", "unknown")))


def adapt_belebele(row: Row, idx: int) -> Optional[Sample]:
    gold = answer_key_to_index(row.get("correct_answer_num"), 4, base=1)
    if gold is None:
        return None
    sample = _mcq(str(row.get("link", idx)) + f"#{idx}", row.get("question", ""),
                  [row.get(f"mc_answer{i}") for i in (1, 2, 3, 4)], gold)
    if sample:
        sample.context = str(row.get("flores_passage", ""))
    return sample


def adapt_piqa(row: Row, idx: int) -> Optional[Sample]:
    gold = answer_key_to_index(row.get("label"), 2, base=0)   # 0/1
    if gold is None:
        return None
    return _mcq(str(idx), row.get("prompt", ""),
                [row.get("solution0"), row.get("solution1")], gold)


def adapt_include(row: Row, idx: int) -> Optional[Sample]:
    options = [row.get(f"option_{c}") for c in ("a", "b", "c", "d")]
    gold = answer_key_to_index(row.get("answer"), 4, base=0)   # 0-indexed
    if gold is None:
        return None
    return _mcq(str(idx), row.get("question", ""), options, gold,
                group=str(row.get("subject", "unknown")))


def adapt_kkcopa(row: Row, idx: int) -> Optional[Sample]:
    gold = answer_key_to_index(row.get("correct_answers"), 2, base=1)
    if gold is None:
        return None
    sample = _mcq(str(idx), row.get("p_contents_kz", ""),
                  [row.get("a1_contents_kz"), row.get("a2_contents_kz")], gold)
    if sample:
        sample.meta["asks_for"] = row.get("asks-for_kz", "")
    return sample


def adapt_nis_math(row: Row, idx: int) -> Optional[Sample]:
    gold = answer_key_to_index(row.get("correct"), 4)   # letters A-D
    if gold is None:
        return None
    return _mcq(str(idx), row.get("question", ""),
                [row.get("a"), row.get("b"), row.get("c"), row.get("d")], gold)


def adapt_kkwikispell(row: Row, idx: int) -> Optional[Sample]:
    mistake = row.get("Sentence with Mistake")
    original = row.get("Original Sentence")
    if not mistake or not original:
        return None
    return Sample(uid=str(idx), question=str(mistake),
                  reference=str(original).strip())


# --- Judged, free-form sets ------------------------------------------------
def adapt_kazqad(row: Row, idx: int) -> Optional[Sample]:
    answers = row.get("answers")
    if isinstance(answers, str):
        answers = parse_dict(answers)
    texts = (answers or {}).get("text") or []
    if not texts:
        return None
    return Sample(uid=str(row.get("id", idx)),
                  question=str(row.get("question", "")),
                  context=str(row.get("context", "")),
                  reference=str(texts[0]))


def adapt_ragbench(row: Row, idx: int) -> Optional[Sample]:
    reference = row.get("response")
    if not reference:
        return None
    return Sample(uid=str(row.get("batch_key", idx)),
                  question=str(row.get("question", "")),
                  documents=join_documents(row.get("documents")),
                  reference=str(reference))


def adapt_ifbench(row: Row, idx: int) -> Optional[Sample]:
    instruction = row.get("instruction_kk")
    if not instruction:
        return None
    constraints = [c for c in (row.get("llm_constraints_kk"),
                               row.get("code_constraints_kk")) if c]
    return Sample(uid=str(row.get("id", idx)),
                  question=str(instruction),
                  constraints=" | ".join(str(c) for c in constraints),
                  reference=None)


# ---------------------------------------------------------------------------
# KazMMLU subsets — discovered from the Hub, with this list as the fallback.
# ---------------------------------------------------------------------------

KAZMMLU_SUBSETS = [
    "Accounting and Auditing (Professional & University in rus)",
    "Biology (High School in kaz)", "Biology (High School in rus)",
    "Biology (Professional & University in rus)",
    "Chemistry (High School in kaz)", "Chemistry (High School in rus)",
    "Culture and Art (Professional & University in rus)",
    "Economics and Entrepreneurship (Professional in rus)",
    "Education and Training (Professional & University in rus)",
    "Finance (Professional & University in rus)",
    "General Education Disciplines (Professional & University in rus)",
    "Geography (High School in kaz)", "Geography (High School in rus)",
    "Informatics (High School in kaz)", "Informatics (High School in rus)",
    "Jurisprudence (Professional & University in rus)",
    "Kazakh History (High School in kaz)", "Kazakh History (High School in rus)",
    "Kazakh Language (High School in kaz)", "Kazakh Literature (High School in kaz)",
    "Law (High School in kaz)", "Law (High School in rus)",
    "Management and Marketing (Professional & University in rus)",
    "Math (High School in kaz)", "Math (High School in rus)",
    "Math Literacy (High School in rus)",
    "Medicine (Professional & University in rus)",
    "Philosophy and Psychology (Professional & University in rus)",
    "Physics (High School in kaz)", "Physics (High School in rus)",
    "Reading Literacy (High School in kaz)", "Reading Literacy (High School in rus)",
    "Russian Language (High School in rus)", "Russian Literature (High School in rus)",
    "Social Science (Professional & University in rus)",
    "World History (High School in kaz)", "World History (High School in rus)",
]

MMLU_REDUX_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

register(BenchmarkSpec(
    key="mmlu",
    task=TaskType.MCQ,
    description="MMLU — translated Kazakh set vs. the original English set.",
    align=True,
    group_name="subject",
    sources={
        "kk": Source("issai/MMLU_Kazakh", adapt_mmlu, config="default",
                     split="test", public=False, uid_counter=True),
        "en": Source("cais/mmlu", adapt_mmlu, config="all", split="test",
                     uid_counter=True,
                     note="aligned to the Kazakh set per subject by position"),
    },
))

register(BenchmarkSpec(
    key="mmlu_pro",
    task=TaskType.MCQ,
    description="MMLU-Pro (10 options) in Kazakh, Russian and English.",
    align=True,
    group_name="category",
    sources={
        "kk": Source("issai/MMLU-Pro_Kazakh_Russian", adapt_mmlu_pro,
                     config="kazakh", split="test", public=False),
        "ru": Source("issai/MMLU-Pro_Kazakh_Russian", adapt_mmlu_pro,
                     config="russian", split="test", public=False),
        "en": Source("TIGER-Lab/MMLU-Pro", adapt_mmlu_pro, config="default",
                     split="test", note="joined on question_id"),
    },
))

register(BenchmarkSpec(
    key="gpqa",
    task=TaskType.MCQ,
    description="GPQA (main + extended + diamond) in Kazakh, Russian, English.",
    max_options=4,
    group_name="subdomain",
    sources={
        "kk": Source("issai/GPQA_Kazakh_Russian", adapt_gpqa_translated,
                     config="kazakh", split="test", public=False),
        "ru": Source("issai/GPQA_Kazakh_Russian", adapt_gpqa_translated,
                     config="russian", split="test", public=False),
        "en": Source("Idavidrein/gpqa", adapt_gpqa_original,
                     configs=["gpqa_main", "gpqa_extended", "gpqa_diamond"],
                     split="train", public=False,
                     note="same three subsets as the translation; gated on the Hub"),
    },
))

register(BenchmarkSpec(
    key="arc",
    task=TaskType.MCQ,
    description="ARC Easy + Challenge in Kazakh, Russian and English.",
    align=True,
    group_name="source",
    sources={
        "kk": Source("issai/ARC_Kazakh_Russian", adapt_arc, config="kazakh",
                     split="test", public=False),
        "ru": Source("issai/ARC_Kazakh_Russian", adapt_arc, config="russian",
                     split="test", public=False),
        "en": Source("allenai/ai2_arc", adapt_arc,
                     configs=["ARC-Easy", "ARC-Challenge"], split="test",
                     group_map={"ARC-Easy": "easy", "ARC-Challenge": "challenge"},
                     note="joined on the ARC item id"),
    },
))

register(BenchmarkSpec(
    key="gsm8k",
    task=TaskType.MATH_NUMERIC,
    description="GSM8K grade-school math in Kazakh, Russian and English.",
    sources={
        "kk": Source("issai/GSM8k_Kazakh_Russian", adapt_gsm8k, config="kazakh",
                     split="test", public=False),
        "ru": Source("issai/GSM8k_Kazakh_Russian", adapt_gsm8k, config="russian",
                     split="test", public=False),
        "en": Source("openai/gsm8k", adapt_gsm8k, config="main", split="test",
                     note="same 1319-item test split as the translation"),
    },
))

register(BenchmarkSpec(
    key="mmlu_redux",
    task=TaskType.MCQ,
    description="MMLU-Redux 2.0 (error-corrected MMLU) in Kazakh and English.",
    align=True,
    group_name="subject",
    sources={
        "kk": Source("issai/MMLU_Redux_2.0_Kazakh", adapt_mmlu_redux_kk,
                     config="default", split="test", public=False),
        "en": Source("edinburgh-dawg/mmlu-redux-2.0", adapt_mmlu_redux_en,
                     configs=MMLU_REDUX_SUBJECTS, split="test",
                     uid_counter=True,
                     note="joined on subject#position"),
    },
))

register(BenchmarkSpec(
    key="polymath",
    task=TaskType.MATH_SYMBOLIC,
    description="PolyMath competition math, graded symbolically with sympy.",
    align=True,
    group_name="difficulty",
    sources={
        "kk": Source("issai/PolyMath_Kazakh", adapt_polymath_kk,
                     config="default", split="test", public=False),
        # PolyMath stores difficulty as the split name, so all four are read.
        "en": Source("Qwen/PolyMath", adapt_polymath_en, config="en",
                     splits=["top", "high", "medium", "low"],
                     note="joined on difficulty-index"),
    },
))

register(BenchmarkSpec(
    key="kazculture",
    task=TaskType.MCQ,
    description="KazCulture — Kazakh cultural knowledge.",
    max_options=4,
    sources={
        "kk": Source("issai/KazCulture", adapt_kazculture, config="default",
                     split="test", public=False),
    },
))

register(BenchmarkSpec(
    key="kazmmlu",
    task=TaskType.MCQ,
    description="KazMMLU — Kazakh/Russian school and professional exams.",
    max_options=5,
    group_name="subset",
    sources={
        "kk": Source("MBZUAI/KazMMLU", adapt_kazmmlu, configs=KAZMMLU_SUBSETS,
                     split="test", configs_from_hub=True),
    },
))

register(BenchmarkSpec(
    key="kazbench_kk",
    task=TaskType.MCQ,
    description="kk-socio-cultural-bench-mc — Kazakh socio-cultural MCQ.",
    max_options=4,
    group_name="category",
    sources={
        "kk": Source("kz-transformers/kk-socio-cultural-bench-mc",
                     adapt_kazbench_kk, config="default", split="train"),
    },
))

register(BenchmarkSpec(
    key="belebele",
    task=TaskType.MCQ,
    description="Belebele reading comprehension (Kazakh subset).",
    max_options=4,
    sources={
        "kk": Source("facebook/belebele", adapt_belebele, config="kaz_Cyrl",
                     split="test"),
    },
))

register(BenchmarkSpec(
    key="piqa",
    task=TaskType.MCQ,
    description="Global-PIQA physical commonsense (Kazakh subset).",
    max_options=2,
    sources={
        "kk": Source("mrlbenchmarks/global-piqa-nonparallel", adapt_piqa,
                     config="kaz_cyrl", split="test"),
    },
))

register(BenchmarkSpec(
    key="include",
    task=TaskType.MCQ,
    description="INCLUDE regional exams — Kazakh and Russian (no English set).",
    max_options=4,
    group_name="subject",
    sources={
        "kk": Source("CohereLabs/include-base-44", adapt_include,
                     config="Kazakh", split="test"),
        "ru": Source("CohereLabs/include-base-44", adapt_include,
                     config="Russian", split="test",
                     note="INCLUDE is non-parallel across languages"),
    },
))

register(BenchmarkSpec(
    key="kkcopa",
    task=TaskType.MCQ,
    description="kkCOPA causal reasoning (local JSONL).",
    max_options=2,
    sources={
        "kk": Source("kkCOPA.jsonl", adapt_kkcopa, kind="jsonl"),
    },
))

register(BenchmarkSpec(
    key="nis_math",
    task=TaskType.MCQ,
    description="NIS Math olympiad-style MCQ (local JSONL).",
    max_options=4,
    sources={
        "kk": Source("NIS_Math.jsonl", adapt_nis_math, kind="jsonl"),
    },
))

register(BenchmarkSpec(
    key="kkwikispell",
    task=TaskType.SPELLING,
    description="kkWikiSpell sentence correction, exact match (local JSONL).",
    sources={
        "kk": Source("kkWikiSpell.jsonl", adapt_kkwikispell, kind="jsonl"),
    },
))

register(BenchmarkSpec(
    key="kazqad",
    task=TaskType.QA_JUDGE,
    description="KazQAD extractive QA — judged 0/1 against the gold answer.",
    sources={
        "kk": Source("issai/kazqad", adapt_kazqad, config="kazqad",
                     split="test", public=False),
    },
))

register(BenchmarkSpec(
    key="ragbench",
    task=TaskType.RAG_JUDGE,
    description="RAGBench grounded generation — judged 0/1 against the gold response.",
    sources={
        "kk": Source("issai/RAGBench_Kazakh", adapt_ragbench, config="default",
                     split="test", public=False),
    },
))

register(BenchmarkSpec(
    key="ifbench",
    task=TaskType.IF_JUDGE,
    description="IFBench constraint following — judged 0/1 against the constraints.",
    sources={
        "kk": Source("issai/IFBench_Kazakh", adapt_ifbench, config="default",
                     split="test", public=False),
    },
))

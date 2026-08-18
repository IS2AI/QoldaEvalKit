"""Prompt construction.

Scaffolding (labels and the answer-format instruction) is written in the
benchmark's own language by default, so a Kazakh item is presented as a fully
Kazakh prompt.  ``--prompt_lang en`` forces English scaffolding instead, which
is useful when comparing against evaluations that prompt in English throughout.
"""

from __future__ import annotations

from typing import Dict

from core.parsing import LETTERS
from core.registry import Sample, TaskType

# The JSON key and its placeholder stay in English in every language: they are
# a format specification, not prose the model should translate.
_T: Dict[str, Dict[str, str]] = {
    "en": {
        "context": "Context",
        "question": "Question",
        "options": "Options",
        "problem": "Problem",
        "documents": "Documents",
        "sentence": "Sentence",
        "mcq_instruction":
            'Answer with the letter of the correct option. '
            'Provide your answer as JSON: {"answer": "LETTER"}',
        "numeric_instruction":
            'Solve the problem step by step. '
            'Provide your final answer as JSON: {"answer": NUMBER}',
        "symbolic_instruction":
            'Solve the problem step by step. '
            'Put your final answer inside \\boxed{}.',
        "qa_instruction":
            'Answer the question using only the context above. '
            'Provide your answer as JSON: {"answer": "your answer"}',
        "rag_instruction":
            'Answer the question using only the documents above.',
        "spelling_instruction":
            'The sentence below contains spelling or grammatical errors. '
            'Return ONLY the corrected sentence, with no explanation.',
    },
    "ru": {
        "context": "Контекст",
        "question": "Вопрос",
        "options": "Варианты",
        "problem": "Задача",
        "documents": "Документы",
        "sentence": "Предложение",
        "mcq_instruction":
            'Ответьте буквой правильного варианта. '
            'Дайте ответ в формате JSON: {"answer": "LETTER"}',
        "numeric_instruction":
            'Решите задачу шаг за шагом. '
            'Дайте окончательный ответ в формате JSON: {"answer": NUMBER}',
        "symbolic_instruction":
            'Решите задачу шаг за шагом. '
            'Поместите окончательный ответ в \\boxed{}.',
        "qa_instruction":
            'Ответьте на вопрос, опираясь только на приведённый выше контекст. '
            'Дайте ответ в формате JSON: {"answer": "ваш ответ"}',
        "rag_instruction":
            'Ответьте на вопрос, опираясь только на приведённые выше документы.',
        "spelling_instruction":
            'В предложении ниже есть орфографические или грамматические ошибки. '
            'Верните ТОЛЬКО исправленное предложение, без пояснений.',
    },
    "kk": {
        "context": "Мәтін",
        "question": "Сұрақ",
        "options": "Нұсқалар",
        "problem": "Есеп",
        "documents": "Құжаттар",
        "sentence": "Сөйлем",
        "mcq_instruction":
            'Дұрыс нұсқаның әрпін жазыңыз. '
            'Жауабыңызды JSON форматында беріңіз: {"answer": "LETTER"}',
        "numeric_instruction":
            'Есепті қадам-қадаммен шығарыңыз. '
            'Соңғы жауабыңызды JSON форматында беріңіз: {"answer": NUMBER}',
        "symbolic_instruction":
            'Есепті қадам-қадаммен шығарыңыз. '
            'Соңғы жауабыңызды \\boxed{} ішіне жазыңыз.',
        "qa_instruction":
            'Тек жоғарыдағы мәтінге сүйене отырып сұраққа жауап беріңіз. '
            'Жауабыңызды JSON форматында беріңіз: {"answer": "жауабыңыз"}',
        "rag_instruction":
            'Тек жоғарыдағы құжаттарға сүйене отырып сұраққа жауап беріңіз.',
        "spelling_instruction":
            'Төмендегі сөйлемде емле немесе грамматикалық қате бар. '
            'ТЕК түзетілген сөйлемді қайтарыңыз, ешқандай түсініктеме бермеңіз.',
    },
}


def _labels(lang: str, prompt_lang: str) -> Dict[str, str]:
    if prompt_lang != "auto":
        lang = prompt_lang
    return _T.get(lang, _T["en"])


def build_prompt(task: TaskType, sample: Sample, lang: str,
                 prompt_lang: str = "auto") -> str:
    """Render a sample into the user-turn text for the model."""
    t = _labels(lang, prompt_lang)

    if task == TaskType.MCQ:
        parts = []
        if sample.context:
            parts.append(f"{t['context']}: {sample.context}\n")
        parts.append(f"{t['question']}: {sample.question}\n")
        parts.append(f"{t['options']}:")
        for i, option in enumerate(sample.options or []):
            parts.append(f"{LETTERS[i]}: {option}")
        parts.append(f"\n{t['mcq_instruction']}")
        return "\n".join(parts)

    if task == TaskType.MATH_NUMERIC:
        return f"{t['problem']}: {sample.question}\n\n{t['numeric_instruction']}"

    if task == TaskType.MATH_SYMBOLIC:
        return f"{t['problem']}: {sample.question}\n\n{t['symbolic_instruction']}"

    if task == TaskType.SPELLING:
        return (f"{t['spelling_instruction']}\n\n"
                f"{t['sentence']}: {sample.question}")

    if task == TaskType.QA_JUDGE:
        return (f"{t['context']}:\n{sample.context}\n\n"
                f"{t['question']}: {sample.question}\n\n"
                f"{t['qa_instruction']}")

    if task == TaskType.RAG_JUDGE:
        return (f"{t['documents']}:\n{sample.documents}\n\n"
                f"{t['question']}: {sample.question}\n\n"
                f"{t['rag_instruction']}")

    if task == TaskType.IF_JUDGE:
        # Instruction-following items are sent verbatim: any scaffolding we add
        # would itself become a constraint the model has to reconcile.
        return sample.question

    raise ValueError(f"No prompt template for task {task}")

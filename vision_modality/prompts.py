"""Vision prompt construction.

Scaffolding is written in the benchmark's own language by default; the image
itself is attached by the client, so these templates only produce the text turn.

RealWorldQA and MMStar already carry their labelled options inside the question
text, so the Options block is emitted only when the sample actually has an
options list.
"""

from __future__ import annotations

from typing import Dict

from core.parsing import LETTERS
from core.registry import Sample, TaskType

_T: Dict[str, Dict[str, str]] = {
    "en": {
        "hint": "Hint",
        "question": "Question",
        "options": "Options",
        "look": "Look at the image and answer the question.",
        "mcq_instruction":
            'Answer with the letter of the correct option. '
            'Provide your answer as JSON: {"answer": "LETTER"}',
        "flexible_instruction":
            'Provide your answer as JSON: {"answer": "LETTER or value"}',
        "math_mcq_instruction":
            'Reason step by step, then answer with the letter of the correct '
            'option. Provide your answer as JSON: {"answer": "LETTER"}',
        "math_open_instruction":
            'Reason step by step. Give only the final value, with no units and '
            'no explanation. Provide your answer as JSON: {"answer": "value"}',
        "ocr_instruction":
            'Read the text in the image. Return only the text you read, with '
            'no explanation. Provide your answer as JSON: {"answer": "text"}',
    },
    "ru": {
        "hint": "Подсказка",
        "question": "Вопрос",
        "options": "Варианты",
        "look": "Посмотрите на изображение и ответьте на вопрос.",
        "mcq_instruction":
            'Ответьте буквой правильного варианта. '
            'Дайте ответ в формате JSON: {"answer": "LETTER"}',
        "flexible_instruction":
            'Дайте ответ в формате JSON: {"answer": "LETTER or value"}',
        "math_mcq_instruction":
            'Рассуждайте шаг за шагом, затем ответьте буквой правильного '
            'варианта. Дайте ответ в формате JSON: {"answer": "LETTER"}',
        "math_open_instruction":
            'Рассуждайте шаг за шагом. Укажите только итоговое значение, без '
            'единиц измерения и пояснений. '
            'Дайте ответ в формате JSON: {"answer": "value"}',
        "ocr_instruction":
            'Прочитайте текст на изображении. Верните только прочитанный '
            'текст, без пояснений. '
            'Дайте ответ в формате JSON: {"answer": "text"}',
    },
    "kk": {
        "hint": "Нұсқау",
        "question": "Сұрақ",
        "options": "Нұсқалар",
        "look": "Суретке қарап, сұраққа жауап беріңіз.",
        "mcq_instruction":
            'Дұрыс нұсқаның әрпін жазыңыз. '
            'Жауабыңызды JSON форматында беріңіз: {"answer": "LETTER"}',
        "flexible_instruction":
            'Жауабыңызды JSON форматында беріңіз: {"answer": "LETTER or value"}',
        "math_mcq_instruction":
            'Қадам-қадаммен ойланып, дұрыс нұсқаның әрпін жазыңыз. '
            'Жауабыңызды JSON форматында беріңіз: {"answer": "LETTER"}',
        "math_open_instruction":
            'Қадам-қадаммен ойланыңыз. Тек соңғы мәнді жазыңыз, өлшем бірлігі '
            'мен түсініктемесіз. '
            'Жауабыңызды JSON форматында беріңіз: {"answer": "value"}',
        "ocr_instruction":
            'Суреттегі мәтінді оқыңыз. Тек оқыған мәтінді қайтарыңыз, '
            'ешқандай түсініктеме бермеңіз. '
            'Жауабыңызды JSON форматында беріңіз: {"answer": "text"}',
    },
}


def _labels(lang: str, prompt_lang: str) -> Dict[str, str]:
    if prompt_lang != "auto":
        lang = prompt_lang
    return _T.get(lang, _T["en"])


def _body(t: Dict[str, str], sample: Sample) -> str:
    parts = []
    if sample.hint:
        parts.append(f"{t['hint']}: {sample.hint}\n")
    parts.append(f"{t['question']}: {sample.question}")
    if sample.options:
        parts.append("")
        parts.append(f"{t['options']}:")
        for i, option in enumerate(sample.options):
            parts.append(f"{LETTERS[i]}: {option}")
    return "\n".join(parts)


def build_prompt(task: TaskType, sample: Sample, lang: str,
                 prompt_lang: str = "auto") -> str:
    """Render a sample into the text half of the multimodal user turn."""
    t = _labels(lang, prompt_lang)
    body = _body(t, sample)

    if task in (TaskType.OCR_MATCH, TaskType.OCR_JUDGE):
        return f"{body}\n\n{t['ocr_instruction']}"

    if task == TaskType.MATH_MIXED:
        instruction = (t["math_mcq_instruction"] if sample.options
                       else t["math_open_instruction"])
        return f"{body}\n\n{instruction}"

    if task == TaskType.MCQ:
        return f"{body}\n\n{t['look']} {t['mcq_instruction']}"

    if task == TaskType.BABY_MIXED:
        instruction = (t["mcq_instruction"] if sample.options
                       else t["flexible_instruction"])
        return f"{body}\n\n{t['look']} {instruction}"

    if task == TaskType.FLEXIBLE:
        return f"{body}\n\n{t['look']} {t['flexible_instruction']}"

    raise ValueError(f"No vision prompt template for task {task}")

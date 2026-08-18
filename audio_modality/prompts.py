"""Audio prompt construction.

SAKURA and SpokenMQA carry their own instruction text in the dataset, already
written in the benchmark's language, so those are used verbatim and only the
answer-format line is appended.  WavCaps and ASR are prompted from scratch.
"""

from __future__ import annotations

from typing import Dict

from core.registry import Sample, TaskType

_T: Dict[str, Dict[str, str]] = {
    "en": {
        "letter_only":
            "Answer with exactly one lowercase letter (a, b, c or d) and "
            "nothing else. No explanation, no punctuation.",
        "numeric":
            'Solve the problem. Give only the final number. '
            'Provide your answer as JSON: {"answer": NUMBER}',
        "caption":
            "Describe only the sounds you actually hear, in one concise "
            "sentence in English. Do not speculate beyond the audio.",
        "audio_qa":
            "Answer the question from the audio alone, in one concise "
            "sentence in English.",
        "asr":
            'Transcribe the speech exactly as spoken, in the original '
            'language. Return only the transcription. '
            'Provide your answer as JSON: {"answer": "transcription"}',
    },
    "ru": {
        "letter_only":
            "Ответьте ровно одной строчной буквой (a, b, c или d) и ничем "
            "больше. Без пояснений и знаков препинания.",
        "numeric":
            'Решите задачу. Укажите только итоговое число. '
            'Дайте ответ в формате JSON: {"answer": NUMBER}',
        "caption":
            "Опишите только те звуки, которые вы действительно слышите, одним "
            "кратким предложением на русском языке. Не домысливайте.",
        "audio_qa":
            "Ответьте на вопрос, опираясь только на аудио, одним кратким "
            "предложением на русском языке.",
        "asr":
            'Запишите произнесённую речь дословно, на языке оригинала. '
            'Верните только транскрипцию. '
            'Дайте ответ в формате JSON: {"answer": "transcription"}',
    },
    "kk": {
        "letter_only":
            "Тек бір кіші әріппен жауап беріңіз (a, b, c немесе d), басқа "
            "ештеңе жазбаңыз. Түсініктеме де, тыныс белгісі де қажет емес.",
        "numeric":
            'Есепті шығарыңыз. Тек соңғы санды жазыңыз. '
            'Жауабыңызды JSON форматында беріңіз: {"answer": NUMBER}',
        "caption":
            "Тек өзіңіз шын естіген дыбыстарды қазақ тілінде бір қысқа "
            "сөйлеммен сипаттаңыз. Аудиода жоқ нәрсені болжамаңыз.",
        "audio_qa":
            "Тек аудиоға сүйене отырып, сұраққа қазақ тілінде бір қысқа "
            "сөйлеммен жауап беріңіз.",
        "asr":
            'Айтылған сөзді түпнұсқа тілінде дәлме-дәл жазып алыңыз. '
            'Тек транскрипцияны қайтарыңыз. '
            'Жауабыңызды JSON форматында беріңіз: {"answer": "transcription"}',
    },
}


def _labels(lang: str, prompt_lang: str) -> Dict[str, str]:
    if prompt_lang != "auto":
        lang = prompt_lang
    return _T.get(lang, _T["en"])


def build_prompt(task: TaskType, sample: Sample, lang: str,
                 prompt_lang: str = "auto") -> str:
    """Render a sample into the text half of the audio user turn."""
    t = _labels(lang, prompt_lang)

    if task == TaskType.MCQ:
        # SAKURA's instruction already lists the options in the right language.
        return f"{sample.question}\n\n{t['letter_only']}"

    if task == TaskType.MATH_NUMERIC:
        return f"{sample.question}\n\n{t['numeric']}"

    if task == TaskType.ASR_WER:
        return t["asr"]

    if task == TaskType.AUDIO_JUDGE:
        instruction = (sample.question or "").strip()
        # WavCaps asks for a caption, WavCaps QA asks a question about the clip.
        guidance = t["audio_qa"] if instruction.endswith("?") else t["caption"]
        return f"{instruction}\n\n{guidance}" if instruction else guidance

    raise ValueError(f"No audio prompt template for task {task}")

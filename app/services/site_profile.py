from __future__ import annotations

import logging
from dataclasses import dataclass
from textwrap import dedent

from openai import AsyncOpenAI

from app.config import settings
from app.services.embeddings import (
    VECTOR_DIM,
    make_embedding_or_none,
    validate_dim,
)

log = logging.getLogger("services.site_profile")

_SYSTEM_PROMPT = (
    "Ты аналитик, который пишет объективные обзоры компаний строго по фактам из "
    "входного текста. Отвечай только в формате одной строки вида "
    "[DESCRIPTION]=[...]. Внутри внутренних скобок располагай структурированный "
    "обзор на русском языке по заданным разделам. Не добавляй пояснений вне "
    "структуры."
)

_INSTRUCTION_BLOCK = dedent(
    """
    [DESCRIPTION]=[САМЫЙ ПОДРОБНЫЙ обзор компании на РУССКОМ ЯЗЫКЕ, только по фактам из исходного текста.
    Допускается много абзацев и списков внутри этих скобок.
    Структурируй по разделам (используй подзаголовки внутри скобок):
    1) Профиль и специализация (кто, чем занимается; сфера/нишa)
    2) Основные продукты и услуги (что именно, ключевые линейки/модули/форматы)
    3) Технологии, процессы, оборудование (конкретика: методы, стандарты, ПО, станки/линии и т.п.; только то, что явно указано)
    4) Отрасли/клиенты/кейсы/география (если названы)
    5) Сертификаты, стандарты качества, соответствия (если упомянуты)
    6) Логистика, сервис, гарантии, условия сотрудничества (если упомянуты)
    7) Уникальные особенности/преимущества (ТОЛЬКО если прямо следуют из текста)
    Если каких-то фактов в тексте нет — раздел включи, но явно укажи «не указано».
    Не добавляй ни одного факта, которого нет в исходном тексте. Без маркетинговых клише и домыслов.]
    """
).strip()

_client: AsyncOpenAI | None = (
    AsyncOpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
)


@dataclass
class SiteProfileResult:
    """Результат генерации описания."""

    prompt: str
    description: str
    model: str


def build_prompt(source_text: str) -> str:
    """Собирает текст пользовательского промпта для модели."""

    cleaned = source_text.strip()
    return (
        "Используй инструкцию ниже и заполни содержимое в квадратных скобках по входному тексту.\n"
        f"{_INSTRUCTION_BLOCK}\n\n"
        f"Исходный текст сайта:\n{cleaned}"
    )


async def generate_site_profile(
    source_text: str, *, chat_model: str | None = None
) -> SiteProfileResult:
    """Формирует описание компании через OpenAI."""

    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not configured")

    if _client is None:
        raise RuntimeError("OpenAI client is not initialized")

    prompt = build_prompt(source_text)
    model_name = chat_model or settings.CHAT_MODEL

    try:
        response = await _client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as exc:  # pragma: no cover - сеть/библиотека
        log.exception("Failed to call OpenAI chat: %s", exc)
        raise RuntimeError("OpenAI chat completion failed") from exc

    if not response.choices:
        raise RuntimeError("OpenAI returned no choices")

    message = response.choices[0].message
    description = (getattr(message, "content", None) or "").strip()
    if not description:
        raise RuntimeError("OpenAI chat completion is empty")

    return SiteProfileResult(prompt=prompt, description=description, model=model_name)


async def build_description_vector(description: str, *, timeout: float = 12.0) -> list[float]:
    """Получает эмбеддинг описания с учётом внутренних/внешних провайдеров."""

    vector = await make_embedding_or_none(description, timeout=timeout)
    if not vector:
        raise RuntimeError("Failed to obtain embedding for description")

    expected_dim = int(getattr(settings, "VECTOR_DIM", VECTOR_DIM) or 0)
    if expected_dim and not validate_dim(vector, expected_dim):
        log.warning(
            "Embedding dimension mismatch: expected ~%s, got %s", expected_dim, len(vector)
        )

    return [float(x) for x in vector]

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
    "входного текста. Сохраняй нейтральный тон, избегай маркетинговых клише и "
    "домыслов. Отвечай только готовым описанием на русском языке без "
    "дополнительных пометок, заголовков, списков или служебных фраз."
)

_INSTRUCTION_BLOCK = dedent(
    """
    Составь максимально подробный фактологичный обзор компании на русском языке, используя только сведения из исходного текста.
    Сделай связный абзац из последовательных полных предложений.
    Сохрани порядок тем: профиль и специализация, ключевые продукты и услуги, используемые технологии и процессы,
    отрасли или клиенты, сертификаты и стандарты, условия сервиса и сотрудничества, уникальные особенности.
    Используй точные подписи тем: «Профиль», «Продукты и услуги», «Технологии и процессы», «Клиенты и отрасли»,
    «Сертификаты и стандарты», «Сервис и сотрудничество», «Уникальные особенности».
    Каждое предложение начинай с подписи темы и двоеточия (например, «Профиль: …»).
    Если сведений по теме нет, напиши «<Подпись темы>: информация не указана». Не объединяй несколько тем в одном предложении.
    Указывай только факты, даты и цифры из исходного текста. Не делай выводов, оценок или предположений.
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
    description = _normalize_description(getattr(message, "content", None) or "")
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


def _normalize_description(raw: str) -> str:
    """Удаляет служебные префиксы, если модель всё же их вернула."""

    text = raw.strip()
    prefix = "[DESCRIPTION]="
    if text.startswith(prefix):
        text = text[len(prefix) :].lstrip()

    if text.startswith("[") and text.endswith("]") and "\n" not in text:
        inner = text[1:-1].strip()
        if inner:
            text = inner

    return text

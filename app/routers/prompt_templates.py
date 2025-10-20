"""Маршруты для генерации промптов OpenAI."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Callable, Tuple

from fastapi import APIRouter

from app.schemas.prompt_templates import (
    PromptTemplateEvent,
    PromptTemplateResponse,
    SiteAvailablePromptRequest,
    SiteUnavailablePromptRequest,
)
from app.services.prompt_templates import (
    build_okved_prompt,
    build_site_available_prompt,
)

log = logging.getLogger("api.prompt_templates")


UTC = dt.timezone.utc


def _timing() -> Tuple[dt.datetime, Callable[[], dt.datetime]]:
    started = dt.datetime.now(UTC)

    def finish() -> dt.datetime:
        return dt.datetime.now(UTC)

    return started, finish


def _duration_ms(started: dt.datetime, finished: dt.datetime) -> int:
    return max(0, int((finished - started).total_seconds() * 1000))


def _response(
    *,
    started: dt.datetime,
    finished: dt.datetime,
    prompt: str | None,
    events: list[PromptTemplateEvent],
    error: Exception | None = None,
) -> PromptTemplateResponse:
    duration = _duration_ms(started, finished)
    prompt_len = len(prompt) if prompt else 0
    success = error is None
    error_text = None
    if error is not None:
        error_text = str(error)
        log.warning(
            "[prompts] generation failed duration_ms=%s error=%s", duration, error_text
        )

    return PromptTemplateResponse(
        success=success,
        prompt=prompt if success else None,
        prompt_len=prompt_len,
        started_at=started,
        finished_at=finished,
        duration_ms=duration,
        events=events,
        error=error_text,
    )

router = APIRouter(prefix="/v1/prompts", tags=["analyze-json"])


@router.post(
    "/site-available",
    response_model=PromptTemplateResponse,
    summary="Сформировать промпт на основе текста сайта",
)
async def prompt_for_available_site(
    payload: SiteAvailablePromptRequest,
) -> PromptTemplateResponse:
    """Возвращает промпт с инструкциями для OpenAI, если текст сайта получен."""
    events: list[PromptTemplateEvent] = []
    started, finish = _timing()

    events.append(
        PromptTemplateEvent(
            step="validate_input",
            status="success",
            detail="Текст сайта и атрибуты компании получены",
        )
    )

    try:
        prompt = build_site_available_prompt(
            payload.text_par, payload.company_name, payload.okved
        )
        events.append(
            PromptTemplateEvent(
                step="build_prompt",
                status="success",
                detail=f"Сформирован промпт длиной {len(prompt)} символов",
            )
        )
        finished = finish()
        return _response(
            started=started,
            finished=finished,
            prompt=prompt,
            events=events,
        )
    except Exception as exc:  # pragma: no cover - защитный сценарий
        log.exception("[prompts] failed to build prompt for доступный сайт")
        events.append(
            PromptTemplateEvent(
                step="build_prompt",
                status="error",
                detail=str(exc),
            )
        )
        finished = finish()
        return _response(
            started=started,
            finished=finished,
            prompt=None,
            events=events,
            error=exc,
        )


@router.post(
    "/site-unavailable",
    response_model=PromptTemplateResponse,
    summary="Сформировать промпт по ОКВЭД, если сайт недоступен",
)
async def prompt_for_unavailable_site(
    payload: SiteUnavailablePromptRequest,
) -> PromptTemplateResponse:
    """Возвращает промпт для определения класса производства по ОКВЭД."""
    events: list[PromptTemplateEvent] = []
    started, finish = _timing()

    events.append(
        PromptTemplateEvent(
            step="validate_input",
            status="success",
            detail="Получен ОКВЭД компании",
        )
    )

    try:
        prompt = build_okved_prompt(payload.okved)
        events.append(
            PromptTemplateEvent(
                step="build_prompt",
                status="success",
                detail=f"Сформирован промпт длиной {len(prompt)} символов",
            )
        )
        finished = finish()
        return _response(
            started=started,
            finished=finished,
            prompt=prompt,
            events=events,
        )
    except Exception as exc:  # pragma: no cover - защитный сценарий
        log.exception("[prompts] failed to build prompt для недоступного сайта")
        events.append(
            PromptTemplateEvent(
                step="build_prompt",
                status="error",
                detail=str(exc),
            )
        )
        finished = finish()
        return _response(
            started=started,
            finished=finished,
            prompt=None,
            events=events,
            error=exc,
        )

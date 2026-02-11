"""Маршруты для генерации промптов OpenAI."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Callable, Dict, Optional, Tuple

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
from app.services.analyzer import call_openai, parse_openai_answer
from app.config import settings

log = logging.getLogger("api.prompt_templates")


UTC = dt.timezone.utc


def _timing() -> Tuple[dt.datetime, Callable[[], dt.datetime]]:
    started = dt.datetime.now(UTC)

    def finish() -> dt.datetime:
        return dt.datetime.now(UTC)

    return started, finish


def _duration_ms(started: dt.datetime, finished: dt.datetime) -> int:
    return max(0, int((finished - started).total_seconds() * 1000))


def _event(
    events: list[PromptTemplateEvent],
    *,
    step: str,
    status: str,
    detail: Optional[str] = None,
    started: Optional[dt.datetime] = None,
) -> int:
    finished = dt.datetime.now(UTC)
    duration = _duration_ms(started, finished) if started else None
    events.append(
        PromptTemplateEvent(
            step=step,
            status=status,  # type: ignore[arg-type]
            detail=detail,
            duration_ms=duration,
        )
    )
    return duration or 0


def _response(
    *,
    started: dt.datetime,
    finished: dt.datetime,
    prompt: str | None,
    answer: Optional[str],
    parsed: Optional[Dict[str, Any]],
    prodclass_by_okved: Optional[int],
    events: list[PromptTemplateEvent],
    timings: Dict[str, int],
    chat_model: Optional[str],
    embed_model: Optional[str],
    error: Exception | None = None,
) -> PromptTemplateResponse:
    duration = _duration_ms(started, finished)
    prompt_len = len(prompt) if prompt else 0
    answer_len = len(answer) if answer else 0
    success = error is None
    error_text = None
    if error is not None:
        error_text = str(error)
        log.warning(
            "[prompts] pipeline failed duration_ms=%s error=%s", duration, error_text
        )

    return PromptTemplateResponse(
        success=success,
        prompt=prompt if prompt else None,
        prompt_len=prompt_len,
        answer=answer if answer else None,
        response=answer if answer else None,
        raw_response=answer if answer else None,
        answer_len=answer_len,
        parsed=parsed if parsed else None,
        prodclass_by_okved=prodclass_by_okved,
        prodclass=prodclass_by_okved,
        started_at=started,
        finished_at=finished,
        duration_ms=duration,
        events=events,
        error=error_text,
        timings=timings,
        chat_model=chat_model,
        embed_model=embed_model,
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
    timings: Dict[str, int] = {}
    started, finish = _timing()

    prompt: Optional[str] = None
    answer: Optional[str] = None
    parsed: Optional[Dict[str, Any]] = None

    text_par = (payload.text_par or "").strip()
    if not text_par:
        _event(
            events,
            step="validate_input",
            status="error",
            detail="text_par отсутствует",
        )
        finished = finish()
        return _response(
            started=started,
            finished=finished,
            prompt=None,
            answer=None,
            parsed=None,
            prodclass_by_okved=None,
            events=events,
            timings=timings,
            chat_model=None,
            embed_model=None,
            error=ValueError("text_par is empty"),
        )

    _event(
        events,
        step="validate_input",
        status="success",
        detail="Текст сайта и атрибуты компании получены",
    )

    chat_model = (payload.chat_model or settings.CHAT_MODEL or "").strip() or None
    embed_model = (payload.embed_model or settings.embed_model or "").strip() or None

    try:
        step_started = dt.datetime.now(UTC)
        prompt = build_site_available_prompt(
            text_par, payload.company_name, payload.okved
        )
        duration = _event(
            events,
            step="build_prompt",
            status="success",
            detail=f"Сформирован промпт длиной {len(prompt)} символов",
            started=step_started,
        )
        timings["build_prompt_ms"] = duration
    except Exception as exc:  # pragma: no cover - защитный сценарий
        log.exception("[prompts] failed to build prompt for доступный сайт")
        _event(
            events,
            step="build_prompt",
            status="error",
            detail=str(exc),
        )
        finished = finish()
        return _response(
            started=started,
            finished=finished,
            prompt=None,
            answer=None,
            parsed=None,
            prodclass_by_okved=None,
            events=events,
            timings=timings,
            chat_model=chat_model,
            embed_model=embed_model,
            error=exc,
        )

    if not chat_model:
        error = ValueError("CHAT_MODEL не задан")
        _event(
            events,
            step="select_chat_model",
            status="error",
            detail=str(error),
        )
        finished = finish()
        return _response(
            started=started,
            finished=finished,
            prompt=prompt,
            answer=None,
            parsed=None,
            prodclass_by_okved=None,
            events=events,
            timings=timings,
            chat_model=None,
            embed_model=embed_model,
            error=error,
        )

    try:
        step_started = dt.datetime.now(UTC)
        answer = await call_openai(prompt, chat_model)
        if not answer:
            raise RuntimeError("LLM вернул пустой ответ")
        duration = _event(
            events,
            step="call_openai",
            status="success",
            detail=f"Получен ответ длиной {len(answer)} символов",
            started=step_started,
        )
        timings["openai_ms"] = duration
    except Exception as exc:  # pragma: no cover - защитный сценарий
        log.exception("[prompts] failed to call OpenAI for доступный сайт")
        _event(
            events,
            step="call_openai",
            status="error",
            detail=str(exc),
        )
        finished = finish()
        return _response(
            started=started,
            finished=finished,
            prompt=prompt,
            answer=None,
            parsed=None,
            prodclass_by_okved=None,
            events=events,
            timings=timings,
            chat_model=chat_model,
            embed_model=embed_model,
            error=exc,
        )

    try:
        step_started = dt.datetime.now(UTC)
        parsed = await parse_openai_answer(answer, text_par, embed_model or "")
        duration = _event(
            events,
            step="parse_answer",
            status="success",
            detail=f"Ответ преобразован в структуру с {len(parsed)} полями",
            started=step_started,
        )
        timings["parse_ms"] = duration
    except Exception as exc:  # pragma: no cover - защитный сценарий
        log.exception("[prompts] failed to parse answer for доступный сайт")
        _event(
            events,
            step="parse_answer",
            status="error",
            detail=str(exc),
        )
        finished = finish()
        return _response(
            started=started,
            finished=finished,
            prompt=prompt,
            answer=answer,
            parsed=None,
            prodclass_by_okved=None,
            events=events,
            timings=timings,
            chat_model=chat_model,
            embed_model=embed_model,
            error=exc,
        )

    finished = finish()
    return _response(
        started=started,
        finished=finished,
        prompt=prompt,
        answer=answer,
        parsed=parsed,
        prodclass_by_okved=None,
        events=events,
        timings=timings,
        chat_model=chat_model,
        embed_model=embed_model,
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
    timings: Dict[str, int] = {}
    started, finish = _timing()

    prompt: Optional[str] = None
    answer: Optional[str] = None
    parsed: Optional[Dict[str, Any]] = None
    prodclass_by_okved: Optional[int] = None

    okved = (payload.okved or "").strip()
    if not okved:
        _event(
            events,
            step="validate_input",
            status="error",
            detail="okved отсутствует",
        )
        finished = finish()
        return _response(
            started=started,
            finished=finished,
            prompt=None,
            answer=None,
            parsed=None,
            prodclass_by_okved=None,
            events=events,
            timings=timings,
            chat_model=None,
            embed_model=None,
            error=ValueError("okved is empty"),
        )

    _event(
        events,
        step="validate_input",
        status="success",
        detail="Получен ОКВЭД компании",
    )

    chat_model = (payload.chat_model or settings.CHAT_MODEL or "").strip() or None

    try:
        step_started = dt.datetime.now(UTC)
        prompt = build_okved_prompt(okved)
        duration = _event(
            events,
            step="build_prompt",
            status="success",
            detail=f"Сформирован промпт длиной {len(prompt)} символов",
            started=step_started,
        )
        timings["build_prompt_ms"] = duration
    except Exception as exc:  # pragma: no cover - защитный сценарий
        log.exception("[prompts] failed to build prompt для недоступного сайта")
        _event(
            events,
            step="build_prompt",
            status="error",
            detail=str(exc),
        )
        finished = finish()
        return _response(
            started=started,
            finished=finished,
            prompt=None,
            answer=None,
            parsed=None,
            prodclass_by_okved=None,
            events=events,
            timings=timings,
            chat_model=chat_model,
            embed_model=None,
            error=exc,
        )

    if not chat_model:
        error = ValueError("CHAT_MODEL не задан")
        _event(
            events,
            step="select_chat_model",
            status="error",
            detail=str(error),
        )
        finished = finish()
        return _response(
            started=started,
            finished=finished,
            prompt=prompt,
            answer=None,
            parsed=None,
            prodclass_by_okved=None,
            events=events,
            timings=timings,
            chat_model=None,
            embed_model=None,
            error=error,
        )

    try:
        step_started = dt.datetime.now(UTC)
        answer = await call_openai(prompt, chat_model)
        if not answer:
            raise RuntimeError("LLM вернул пустой ответ")
        duration = _event(
            events,
            step="call_openai",
            status="success",
            detail=f"Получен ответ длиной {len(answer)} символов",
            started=step_started,
        )
        timings["openai_ms"] = duration
    except Exception as exc:  # pragma: no cover - защитный сценарий
        log.exception("[prompts] failed to call OpenAI для недоступного сайта")
        _event(
            events,
            step="call_openai",
            status="error",
            detail=str(exc),
        )
        finished = finish()
        return _response(
            started=started,
            finished=finished,
            prompt=prompt,
            answer=None,
            parsed=None,
            prodclass_by_okved=None,
            events=events,
            timings=timings,
            chat_model=chat_model,
            embed_model=None,
            error=exc,
        )

    try:
        step_started = dt.datetime.now(UTC)
        digits = "".join(ch for ch in answer if ch.isdigit())
        if not digits:
            raise ValueError("Ответ не содержит идентификатора класса")
        prodclass_id = int(digits)
        parsed = {"PRODCLASS": prodclass_id}
        prodclass_by_okved = prodclass_id
        duration = _event(
            events,
            step="parse_answer",
            status="success",
            detail=f"Определён класс производства {prodclass_id}",
            started=step_started,
        )
        timings["parse_ms"] = duration
    except Exception as exc:  # pragma: no cover - защитный сценарий
        log.exception("[prompts] failed to parse answer для недоступного сайта")
        _event(
            events,
            step="parse_answer",
            status="error",
            detail=str(exc),
        )
        finished = finish()
        return _response(
            started=started,
            finished=finished,
            prompt=prompt,
            answer=answer,
            parsed=None,
            prodclass_by_okved=None,
            events=events,
            timings=timings,
            chat_model=chat_model,
            embed_model=None,
            error=exc,
        )

    finished = finish()
    return _response(
        started=started,
        finished=finished,
        prompt=prompt,
        answer=answer,
        parsed=parsed,
        prodclass_by_okved=prodclass_by_okved,
        events=events,
        timings=timings,
        chat_model=chat_model,
        embed_model=None,
    )

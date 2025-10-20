"""Pydantic-схемы для генерации промптов OpenAI."""
from __future__ import annotations

import datetime as dt
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field


class SiteAvailablePromptRequest(BaseModel):
    """Запрос на генерацию промпта с текстом сайта."""

    text_par: str = Field(..., description="Текст, полученный с сайта компании")
    company_name: str = Field(..., description="Название компании")
    okved: str = Field(..., description="ОКВЭД компании")
    chat_model: Optional[str] = Field(
        None,
        description="Название модели OpenAI для генерации ответа (по умолчанию из настроек)",
    )
    embed_model: Optional[str] = Field(
        None,
        description="Название модели эмбеддингов для постобработки (по умолчанию из настроек)",
    )


class SiteUnavailablePromptRequest(BaseModel):
    """Запрос на генерацию промпта, когда сайт недоступен."""

    okved: str = Field(..., description="ОКВЭД компании")
    chat_model: Optional[str] = Field(
        None,
        description="Название модели OpenAI для генерации ответа (по умолчанию из настроек)",
    )


class PromptTemplateEvent(BaseModel):
    """Описание шага обработки запроса на генерацию промпта."""

    step: str = Field(..., description="Короткое имя этапа")
    status: Literal["success", "skipped", "error"] = Field(
        ...,
        description="Статус этапа обработки",
    )
    detail: Optional[str] = Field(
        None,
        description="Дополнительная информация о результате этапа",
    )
    duration_ms: Optional[int] = Field(
        None,
        description="Продолжительность этапа в миллисекундах",
        ge=0,
    )


class PromptTemplateResponse(BaseModel):
    """Ответ с информацией о генерации промпта."""

    success: bool = Field(..., description="Флаг успешного завершения генерации")
    prompt: Optional[str] = Field(
        None,
        description="Текст промпта для OpenAI (если генерация успешна)",
    )
    prompt_len: int = Field(
        ...,
        description="Количество символов в сгенерированном промпте",
        ge=0,
    )
    answer: Optional[str] = Field(
        None,
        description="Ответ OpenAI в сыром виде",
    )
    answer_len: int = Field(
        0,
        description="Количество символов в ответе OpenAI",
        ge=0,
    )
    parsed: Optional[Dict[str, object]] = Field(
        None,
        description="Результат постобработки ответа OpenAI",
    )
    started_at: dt.datetime = Field(
        ..., description="Время начала обработки в формате ISO 8601"
    )
    finished_at: dt.datetime = Field(
        ..., description="Время окончания обработки в формате ISO 8601"
    )
    duration_ms: int = Field(
        ..., description="Время обработки в миллисекундах", ge=0
    )
    events: list[PromptTemplateEvent] = Field(
        default_factory=list,
        description="Хронология этапов обработки запроса",
    )
    error: Optional[str] = Field(
        None,
        description="Текст ошибки, если генерация завершилась неуспешно",
    )
    timings: Dict[str, int] = Field(
        default_factory=dict,
        description="Сводная информация о длительности этапов обработки",
    )
    chat_model: Optional[str] = Field(
        None,
        description="Модель OpenAI, использованная для генерации ответа",
    )
    embed_model: Optional[str] = Field(
        None,
        description="Модель эмбеддингов, использованная при постобработке",
    )

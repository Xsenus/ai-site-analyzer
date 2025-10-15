from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SiteProfileRequest(BaseModel):
    """Запрос на генерацию структурированного описания компании."""

    source_text: str = Field(
        ..., min_length=1, description="Сырый текст с сайта компании"
    )
    chat_model: Optional[str] = Field(
        default=None,
        description="Имя модели OpenAI для генерации описания. По умолчанию settings.CHAT_MODEL.",
    )
    return_prompt: bool = Field(
        default=False,
        description="Вернуть ли использованный промпт в ответе (для отладки).",
    )


class SiteProfileResponse(BaseModel):
    """Ответ с описанием и вектором для записи в БД."""

    description: str = Field(..., description="Готовое описание компании")
    description_vector: List[float] = Field(
        ..., description="Векторное представление описания"
    )
    vector_dim: int = Field(..., ge=1, description="Размерность возвращённого вектора")
    chat_model: str = Field(
        ..., description="Имя модели, использованной для генерации описания"
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Фактический промпт, отправленный в модель (если запрошено)",
    )

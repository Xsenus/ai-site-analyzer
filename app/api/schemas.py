from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field, model_validator


class CatalogVector(BaseModel):
    """Гибкое представление вектора из внешнего сервиса."""

    values: List[float] | None = None
    literal: str | None = None

    def normalized(self) -> tuple[Optional[List[float]], Optional[str]]:
        """Возвращает (значения, текстовое представление) с приведением типов."""

        floats: Optional[List[float]]
        if self.values is None:
            floats = None
        else:
            floats = [float(v) for v in self.values]

        literal = (self.literal or "").strip() or None
        return floats, literal


class CatalogItem(BaseModel):
    id: int
    name: str
    vec: CatalogVector | List[float] | str | None = None


class CatalogItemsPayload(BaseModel):
    """Совместимый контейнер для каталога (list либо объект с items)."""

    items: List[CatalogItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Dict[str, Any]:  # type: ignore[override]
        if value is None:
            return {"items": []}
        if isinstance(value, dict):
            if "items" in value:
                payload = value.get("items")
                if payload is None:
                    return {"items": []}
                return {"items": payload}
            if {"id", "name"} <= set(value.keys()):
                return {"items": [value]}
            raise TypeError("catalog object must contain 'items' list")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return {"items": list(value)}
        raise TypeError("catalog must be list or object with items")


class AnalyzeFromJsonRequest(BaseModel):
    text_par: str = Field(..., description="Сырый текст сайта для анализа")
    pars_id: int | None = Field(default=None, ge=1, description="Опциональный pars_site.id для логов")
    chat_model: str | None = None
    embed_model: str | None = None
    company_id: int | None = Field(default=None, ge=1)
    goods_catalog: CatalogItemsPayload | None = None
    equipment_catalog: CatalogItemsPayload | None = None
    return_prompt: bool = False
    return_answer_raw: bool = True


class VectorPayload(BaseModel):
    values: List[float] | None = None
    literal: str | None = None
    dim: int = 0


class ProdclassPayload(BaseModel):
    id: int
    score: float
    title: str
    score_source: str
    source: str


class MatchedItemPayload(BaseModel):
    text: str
    match_id: int | None = None
    score: float | None = None
    vector: VectorPayload


class RequestCostPayload(BaseModel):
    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cost_usd: float


class BillingSummaryPayload(BaseModel):
    currency: str
    period_start: int
    period_end: int
    spent_usd: float
    limit_usd: float | None = None
    prepaid_credits_usd: float | None = None
    remaining_usd: float | None = None


class DbPayload(BaseModel):
    description: str
    description_vector: VectorPayload
    prodclass: ProdclassPayload
    goods_types: List[MatchedItemPayload]
    equipment: List[MatchedItemPayload]
    llm_answer: str | None = None


class AnalyzeFromJsonResponse(BaseModel):
    pars_id: int | None = None
    prompt: str | None = None
    prompt_len: int
    answer_raw: str | None = None
    answer_len: int
    description: str
    parsed: Dict[str, Any]
    prodclass: ProdclassPayload
    description_vector: VectorPayload
    goods_items: List[MatchedItemPayload]
    equipment_items: List[MatchedItemPayload]
    counts: Dict[str, int]
    timings: Dict[str, int]
    catalogs: Dict[str, int]
    request_cost: RequestCostPayload | None = None
    billing_summary: BillingSummaryPayload | None = None
    db_payload: DbPayload

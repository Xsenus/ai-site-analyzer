from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.equipment_selection import ClientRow, EquipmentSelectionResponse


class AnalyzeRequest(BaseModel):
    chat_model: str | None = Field(default=None)
    embed_model: str | None = Field(default=None)
    dry_run: bool = Field(default=False)
    return_prompt: bool = Field(default=False)
    return_answer_raw: bool = Field(default=True)
    sync_mode: Literal["primary_only", "dual_write", "fallback_to_secondary"] | None = None
    company_id: int | None = Field(default=None, ge=1, description="ID компании для pars_site.company_id")


class AnalyzeResponse(BaseModel):
    pars_id: int
    used_db_mode: str
    prompt_len: int
    answer_len: int
    answer_raw: str | None
    parsed: Dict[str, Any]
    report: Dict[str, Any] | None
    duration_ms: int


class CatalogItem(BaseModel):
    id: int
    name: str
    vec: List[float] | str | None = None


class AnalyzeFromJsonRequest(BaseModel):
    text_par: str = Field(..., description="Сырый текст сайта для анализа")
    pars_id: int | None = Field(default=None, ge=1, description="Опциональный pars_site.id для логов")
    chat_model: str | None = None
    embed_model: str | None = None
    company_id: int | None = Field(default=None, ge=1)
    goods_catalog: List[CatalogItem] | None = None
    equipment_catalog: List[CatalogItem] | None = None
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


class DbPayload(BaseModel):
    description: str
    description_vector: VectorPayload
    prodclass: ProdclassPayload
    goods_types: List[MatchedItemPayload]
    equipment: List[MatchedItemPayload]


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
    db_payload: DbPayload


class IbMatchRequest(BaseModel):
    client_id: int = Field(..., ge=1)
    reembed_if_exists: bool = False
    sync_mode: Literal["primary_only", "dual_write", "fallback_to_secondary"] | None = None


class IbMatchItem(BaseModel):
    ai_id: int
    source_text: str
    match_id: int | None
    match_name: str | None
    score: float | None
    note: str | None = None


class IbMatchSummary(BaseModel):
    goods_total: int
    goods_updated: int
    goods_embedded: int
    equipment_total: int
    equipment_updated: int
    equipment_embedded: int
    catalog_goods_total: int
    catalog_equipment_total: int


class IbMatchResponse(BaseModel):
    client_id: int
    goods: List[IbMatchItem]
    equipment: List[IbMatchItem]
    summary: IbMatchSummary
    debug_report: str | None = None
    duration_ms: int


class ParsSiteInfo(BaseModel):
    id: int
    company_id: Optional[int] = None
    domain_1: Optional[str] = None
    url: Optional[str] = None


class SitePipelineParseResult(BaseModel):
    client: Optional[ClientRow] = None
    pars_site: Optional[ParsSiteInfo] = None
    pars_site_candidates: List[ParsSiteInfo] = Field(default_factory=list)


class SitePipelineResolved(BaseModel):
    requested_inn: Optional[str] = None
    requested_site: Optional[str] = None
    normalized_site: Optional[str] = None
    inn: Optional[str] = None
    pars_site_id: int
    client_id: int


class SitePipelineIbMatchOptions(BaseModel):
    reembed_if_exists: bool = False
    sync_mode: Literal["primary_only", "dual_write", "fallback_to_secondary"] | None = None


class SitePipelineRequest(BaseModel):
    inn: Optional[str] = Field(default=None)
    site: Optional[str] = Field(default=None)
    pars_site_id: Optional[int] = Field(default=None, ge=1)
    client_id: Optional[int] = Field(default=None, ge=1)
    run_analyze: bool = True
    run_ib_match: bool = True
    run_equipment_selection: bool = True
    analyze: AnalyzeRequest | None = None
    ib_match: SitePipelineIbMatchOptions | None = None


class SitePipelineResponse(BaseModel):
    resolved: SitePipelineResolved
    parse_site: SitePipelineParseResult
    analyze: AnalyzeResponse | None = None
    ib_match: IbMatchResponse | None = None
    equipment_selection: EquipmentSelectionResponse | None = None


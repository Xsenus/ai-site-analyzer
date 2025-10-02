from __future__ import annotations
from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    chat_model: str | None = Field(default=None)
    embed_model: str | None = Field(default=None)
    dry_run: bool = Field(default=False)
    return_prompt: bool = Field(default=False)
    return_answer_raw: bool = Field(default=True)
    sync_mode: Literal["primary_only", "dual_write", "fallback_to_secondary"] | None = None


class AnalyzeResponse(BaseModel):
    pars_id: int
    used_db_mode: str
    prompt_len: int
    answer_len: int
    answer_raw: str | None
    parsed: Dict[str, Any]
    report: Dict[str, Any] | None
    duration_ms: int


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
    duration_ms: int

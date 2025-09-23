from __future__ import annotations
from typing import Any, Dict, Literal
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

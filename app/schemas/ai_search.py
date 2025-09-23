# app/schemas/ai_search.py
from __future__ import annotations

from typing import Dict, List, Optional, Annotated
from pydantic import BaseModel, Field


class AiSearchIn(BaseModel):
    # обязательное поле, ограничение по длине
    q: str = Field(..., min_length=1, max_length=2000)


class GoodsRow(BaseModel):
    id: int
    name: str


class EquipRow(BaseModel):
    id: int
    equipment_name: str
    # делаем действительно необязательными (по умолчанию None), чтобы не ловить warning'и типизации
    industry_id: Optional[int] = None
    prodclass_id: Optional[int] = None
    workshop_id: Optional[int] = None


class ProdclassRow(BaseModel):
    id: int
    prodclass: str
    industry_id: Optional[int] = None


class AiEmbeddingOut(BaseModel):
    # pydantic v2: вместо conlist используем Annotated + Field(min_length=1)
    embedding: Annotated[List[float], Field(min_length=1)]


class AiIdsOut(BaseModel):
    ids: Dict[str, List[int]]


class AiListsOut(BaseModel):
    goods: Optional[List[GoodsRow]] = None
    equipment: Optional[List[EquipRow]] = None
    prodclasses: Optional[List[ProdclassRow]] = None


class ErrorOut(BaseModel):
    error: str

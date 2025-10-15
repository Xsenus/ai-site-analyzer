from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.site_profile import SiteProfileRequest, SiteProfileResponse
from app.services.site_profile import (
    SiteProfileResult,
    build_description_vector,
    generate_site_profile,
)

log = logging.getLogger("routers.site_profile")

router = APIRouter(prefix="/v1/site-profile", tags=["site-profile"])


@router.post("", response_model=SiteProfileResponse)
async def create_site_profile(payload: SiteProfileRequest) -> SiteProfileResponse:
    """Генерация подробного описания компании и его эмбеддинга."""

    source_text = payload.source_text.strip()
    if not source_text:
        raise HTTPException(status_code=400, detail="source_text must not be empty")

    try:
        result: SiteProfileResult = await generate_site_profile(
            source_text, chat_model=payload.chat_model
        )
        vector = await build_description_vector(result.description)
    except RuntimeError as exc:
        log.error("site-profile failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    response = SiteProfileResponse(
        description=result.description,
        description_vector=vector,
        vector_dim=len(vector),
        chat_model=result.model,
    )
    if payload.return_prompt:
        response.prompt = result.prompt
    return response

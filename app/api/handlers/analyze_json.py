from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, cast

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    AnalyzeFromJsonRequest,
    AnalyzeFromJsonResponse,
    CatalogItem,
    CatalogItemsPayload,
    CatalogVector,
    MatchedItemPayload,
    DbPayload,
    ProdclassPayload,
    VectorPayload,
)
from app.config import settings
from app.models.ib_prodclass import IB_PRODCLASS
from app.services.analyzer import (
    MATCH_THRESHOLD_EQUIPMENT,
    MATCH_THRESHOLD_GOODS,
    build_prompt,
    call_openai,
    embed_single_text,
    enrich_by_catalog,
    parse_openai_answer,
)
from app.utils.vectors import format_pgvector

log = logging.getLogger("api.analyze_json")

router = APIRouter(prefix="/v1/analyze", tags=["analyze-json"])


def _clip(value: Optional[str], limit: int = 200) -> str:
    if not value:
        return ""
    text = str(value)
    return text if len(text) <= limit else text[:limit] + f"... ({len(text)} chars)"


def _tick(started: dt.datetime) -> int:
    return int((dt.datetime.now() - started).total_seconds() * 1000)


def _as_int(value: Any, *, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, *, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_catalog_vector(
    vec: CatalogVector | Sequence[float] | str | Dict[str, Any] | None,
    *,
    context: str,
) -> Optional[str]:
    values: Optional[List[float]] = None
    literal: Optional[str] = None

    if isinstance(vec, CatalogVector):
        try:
            values, literal = vec.normalized()
        except (TypeError, ValueError):
            log.warning("[analyze/json] catalog vector invalid %s", context)
            values, literal = None, None
    elif isinstance(vec, (list, tuple)):
        try:
            values = [float(v) for v in vec]
        except (TypeError, ValueError):
            log.warning("[analyze/json] catalog vector conversion failed %s", context)
            values = None
    elif isinstance(vec, str):
        literal = vec.strip() or None
    elif isinstance(vec, dict):
        try:
            model = CatalogVector.model_validate(vec)
        except Exception:  # pragma: no cover - pydantic already logs details
            log.warning("[analyze/json] catalog vector dict invalid %s", context)
            return None
        return _normalize_catalog_vector(model, context=context)
    elif vec is not None:
        log.warning("[analyze/json] catalog vector unsupported type %s value=%r", context, vec)
        return None

    if values is None:
        return literal
    if not values:
        return literal or "[]"
    return literal or format_pgvector(values)


def _catalog_items_to_dict(
    items: Optional[CatalogItemsPayload | Sequence[CatalogItem]],
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not items:
        return normalized

    source: Iterable[CatalogItem]
    if isinstance(items, CatalogItemsPayload):
        source = items.items
    else:
        source = items

    for item in source:
        vec_literal = _normalize_catalog_vector(
            item.vec,
            context=f"id={item.id} name={item.name!r}",
        )
        normalized.append({"id": item.id, "name": item.name, "vec": vec_literal})

    return normalized


def _vector_payload(values: Optional[List[float]], literal: Optional[str]) -> VectorPayload:
    safe_values: Optional[List[float]] = None
    if values:
        try:
            safe_values = [float(v) for v in values]
        except (TypeError, ValueError):
            log.warning(
                "[analyze/json] failed to normalize vector values length=%s",
                len(values),
            )
            safe_values = None

    literal_str = (literal.strip() if isinstance(literal, str) else None) or None
    if safe_values and not literal_str:
        literal_str = format_pgvector(safe_values)

    dim = len(safe_values) if safe_values else 0
    return VectorPayload(values=safe_values, literal=literal_str, dim=dim)


def _matched_payload(items: List[Dict[str, Any]]) -> List[MatchedItemPayload]:
    payload: List[MatchedItemPayload] = []
    for it in items:
        vec_values = it.get("vec")
        vec_literal = it.get("vec_str")
        vector = _vector_payload(
            vec_values if isinstance(vec_values, list) else None,
            vec_literal,
        )
        payload.append(
            MatchedItemPayload(
                text=str(it.get("text") or ""),
                match_id=_as_int(it.get("match_id")),
                score=_as_float(it.get("score")),
                vector=vector,
            )
        )
    return payload


def _ai_site_preview(
    items: List[Dict[str, Any]],
    *,
    pars_id: Optional[int],
    name_key: str,
    id_key: str,
    score_key: str,
) -> List[Dict[str, Any]]:
    preview: List[Dict[str, Any]] = []
    for it in items:
        vector_literal_raw = it.get("vec_str")
        vector_literal = None
        if isinstance(vector_literal_raw, str):
            literal = vector_literal_raw.strip()
            vector_literal = literal or None
        vector_dim = 0
        vec_values = it.get("vec")
        if isinstance(vec_values, list):
            try:
                vector_dim = len(vec_values)
            except TypeError:
                vector_dim = 0

        preview.append(
            {
                "text_par_id": pars_id,
                name_key: str(it.get("text") or ""),
                id_key: _as_int(it.get("match_id")),
                score_key: _as_float(it.get("score")),
                "text_vector": vector_literal,
                "vector_dim": vector_dim,
            }
        )
    return preview


@router.post("/json", response_model=AnalyzeFromJsonResponse)
async def analyze_from_json(body: AnalyzeFromJsonRequest) -> AnalyzeFromJsonResponse:
    """Run the analysis pipeline entirely in memory and return structured data."""

    started = dt.datetime.now()
    pars_id = _as_int(body.pars_id)
    text_par = str(body.text_par or "").strip()

    if not text_par:
        log.error("[analyze/json] empty text_par")
        raise HTTPException(status_code=400, detail="text_par is required")

    chat_model_raw = body.chat_model
    embed_model_raw = body.embed_model

    chat_model = (chat_model_raw or settings.CHAT_MODEL or "").strip()
    embed_model = (embed_model_raw or settings.embed_model or "").strip()

    if not chat_model:
        log.error("[analyze/json] chat_model is empty")
        raise HTTPException(status_code=400, detail="CHAT_MODEL не задан")
    if not embed_model:
        log.error("[analyze/json] embed_model is empty")
        raise HTTPException(status_code=400, detail="embed_model не задан")

    goods_catalog = _catalog_items_to_dict(body.goods_catalog)
    equip_catalog = _catalog_items_to_dict(body.equipment_catalog)

    log.info(
        "[analyze/json] start pars_id=%s company_id=%s text_len=%s goods_catalog=%s equipment_catalog=%s",
        pars_id if pars_id is not None else "∅",
        body.company_id,
        len(text_par),
        len(goods_catalog),
        len(equip_catalog),
    )
    log.info(
        "[analyze/json] model sources chat=%s embed=%s",
        "request" if chat_model_raw else "settings",
        "request" if embed_model_raw else "settings",
    )
    log.debug("[analyze/json] text_sample=%r", _clip(text_par))
    log.info(
        "[analyze/json] config chat_model=%s embed_model=%s return_prompt=%s return_answer=%s",
        chat_model,
        embed_model,
        body.return_prompt,
        body.return_answer_raw,
    )
    if goods_catalog or equip_catalog:
        log.debug(
            "[analyze/json] catalogs normalized goods=%r equipment=%r",
            goods_catalog[:3],
            equip_catalog[:3],
        )

    timings: Dict[str, int] = {}

    t_prompt = dt.datetime.now()
    prompt = build_prompt(text_par)
    timings["build_prompt_ms"] = _tick(t_prompt)
    log.info(
        "[analyze/json] prompt built len=%s took_ms=%s",
        len(prompt),
        timings["build_prompt_ms"],
    )

    t_llm = dt.datetime.now()
    answer = await call_openai(prompt, chat_model)
    timings["llm_ms"] = _tick(t_llm)
    log.info(
        "[analyze/json] llm answer len=%s took_ms=%s",
        len(answer or ""),
        timings["llm_ms"],
    )
    log.debug("[analyze/json] llm_answer=%r", _clip(answer))

    if answer is None or not str(answer).strip():
        log.error("[analyze/json] empty llm answer")
        raise HTTPException(status_code=502, detail="LLM вернул пустой ответ")

    t_parse = dt.datetime.now()
    parsed_obj: Any = await parse_openai_answer(answer, text_par, embed_model)
    timings["parse_ms"] = _tick(t_parse)
    if not isinstance(parsed_obj, dict):
        log.error("[analyze/json] parser returned non-dict type=%s", type(parsed_obj).__name__)
        raise HTTPException(
            status_code=502,
            detail="Парсер ответа LLM вернул некорректную структуру",
        )

    parsed: Dict[str, Any] = cast(Dict[str, Any], parsed_obj)
    log.info(
        "[analyze/json] parse complete keys=%s took_ms=%s",
        sorted(parsed.keys()),
        timings["parse_ms"],
    )
    log.debug("[analyze/json] parsed snippet %r", {k: parsed[k] for k in list(parsed)[:5]})

    prompt_payload = prompt if body.return_prompt else None

    prodclass_id = _as_int(parsed.get("PRODCLASS"))
    prodclass_score = _as_float(parsed.get("PRODCLASS_SCORE"))
    score_source = str(parsed.get("PRODCLASS_SCORE_SOURCE") or "model_reply")
    prodclass_source = str(parsed.get("PRODCLASS_SOURCE") or "model_reply")

    if prodclass_id is None:
        log.error("[analyze/json] prodclass missing")
        raise HTTPException(
            status_code=400,
            detail="PRODCLASS отсутствует или не является целым числом",
        )
    if prodclass_score is None:
        log.error("[analyze/json] prodclass_score missing")
        raise HTTPException(
            status_code=400,
            detail="PRODCLASS_SCORE отсутствует или не является числом",
        )

    prodclass_title = IB_PRODCLASS.get(prodclass_id, "—")
    prodclass_payload = ProdclassPayload(
        id=prodclass_id,
        score=float(prodclass_score),
        title=prodclass_title,
        score_source=score_source,
        source=prodclass_source,
    )

    description_text = str(parsed.get("DESCRIPTION") or "").strip()
    description_vector: Optional[List[float]] = None
    description_ms: Optional[int] = None

    if description_text:
        t_desc = dt.datetime.now()
        try:
            description_vector = await embed_single_text(description_text, embed_model)
            description_ms = _tick(t_desc)
            log.info(
                "[analyze/json] description embedding dim=%s took_ms=%s",
                len(description_vector or []),
                description_ms,
            )
        except Exception as exc:  # pragma: no cover - network errors
            description_ms = _tick(t_desc)
            log.error(
                "[analyze/json] description embedding failed took_ms=%s error=%s",
                description_ms,
                exc,
                exc_info=True,
            )
            description_vector = None
    else:
        log.info("[analyze/json] description empty — skip embedding")
    log.debug("[analyze/json] description_text=%r", _clip(description_text))

    timings["description_embed_ms"] = description_ms or 0
    description_payload = _vector_payload(description_vector, None)

    goods_source_list: List[str] = cast(List[str], parsed.get("GOODS_TYPE_LIST", []) or [])
    equip_source_list: List[str] = cast(List[str], parsed.get("EQUIPMENT_LIST", []) or [])
    goods_origin = str(parsed.get("GOODS_TYPE_SOURCE") or "GOODS_TYPE")

    log.info(
        "[analyze/json] sources goods=%s equip=%s goods_origin=%s",
        len(goods_source_list),
        len(equip_source_list),
        goods_origin,
    )
    log.debug(
        "[analyze/json] sources samples goods=%r equip=%r",
        goods_source_list[:5],
        equip_source_list[:5],
    )

    t_enrich_goods = dt.datetime.now()
    goods_enriched = await enrich_by_catalog(
        goods_source_list,
        goods_catalog,
        embed_model,
        MATCH_THRESHOLD_GOODS,
    )
    timings["goods_enrich_ms"] = _tick(t_enrich_goods)

    t_enrich_equip = dt.datetime.now()
    equip_enriched = await enrich_by_catalog(
        equip_source_list,
        equip_catalog,
        embed_model,
        MATCH_THRESHOLD_EQUIPMENT,
    )
    timings["equipment_enrich_ms"] = _tick(t_enrich_equip)

    log.info(
        "[analyze/json] enrichment goods=%s equip=%s goods_ms=%s equip_ms=%s",
        len(goods_enriched),
        len(equip_enriched),
        timings["goods_enrich_ms"],
        timings["equipment_enrich_ms"],
    )
    log.debug("[analyze/json] goods_enriched sample %r", goods_enriched[:3])
    log.debug("[analyze/json] equip_enriched sample %r", equip_enriched[:3])

    goods_payload = _matched_payload(goods_enriched)
    equip_payload = _matched_payload(equip_enriched)

    goods_preview = _ai_site_preview(
        goods_enriched,
        pars_id=pars_id,
        name_key="goods_type",
        id_key="goods_type_ID",
        score_key="goods_types_score",
    )
    equip_preview = _ai_site_preview(
        equip_enriched,
        pars_id=pars_id,
        name_key="equipment",
        id_key="equipment_ID",
        score_key="equipment_score",
    )

    counts = {
        "goods_source": len(goods_source_list),
        "equip_source": len(equip_source_list),
        "goods_enriched": len(goods_payload),
        "equip_enriched": len(equip_payload),
    }

    answer_payload = answer
    answer_raw = answer if body.return_answer_raw else None
    parsed_with_title = parsed | {
        "PRODCLASS_TITLE": prodclass_title,
        "LLM_ANSWER": answer_payload,
        "AI_SITE_GOODS_TYPES": goods_preview,
        "AI_SITE_EQUIPMENT": equip_preview,
    }

    db_payload = DbPayload(
        description=description_text,
        description_vector=description_payload,
        prodclass=prodclass_payload,
        goods_types=goods_payload,
        equipment=equip_payload,
        llm_answer=answer_payload,
    )

    log.info(
        "[analyze/json] db_payload ready description_len=%s goods=%s equipment=%s",
        len(description_text),
        len(goods_payload),
        len(equip_payload),
    )

    timings["total_ms"] = _tick(started)
    log.info(
        "[analyze/json] done pars_id=%s total_ms=%s",
        pars_id if pars_id is not None else "∅",
        timings["total_ms"],
    )

    response = AnalyzeFromJsonResponse(
        pars_id=pars_id,
        prompt=prompt_payload,
        prompt_len=len(prompt),
        answer_raw=answer_raw,
        answer_len=len(answer or ""),
        description=description_text,
        parsed=parsed_with_title,
        prodclass=prodclass_payload,
        description_vector=description_payload,
        goods_items=goods_payload,
        equipment_items=equip_payload,
        counts=counts,
        timings=timings,
        catalogs={"goods": len(goods_catalog), "equipment": len(equip_catalog)},
        db_payload=db_payload,
    )

    log.debug("[analyze/json] response summary %s", response.model_dump(exclude_none=True))

    return response

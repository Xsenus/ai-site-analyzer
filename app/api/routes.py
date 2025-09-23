# app/api/routes.py
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional, Callable, Awaitable, cast

from fastapi import APIRouter, HTTPException
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import settings
from app.api.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.analyzer import (
    build_prompt,
    call_openai,
    parse_openai_answer,
    enrich_by_catalog,
    MATCH_THRESHOLD_EQUIPMENT,
    MATCH_THRESHOLD_GOODS,
)
from app.models.ib_prodclass import IB_PRODCLASS
from app.repositories import parsing_repo as repo
from app.db.tx import (
    SyncMode,
    get_primary_engine,
    get_secondary_engine,
    run_on_engine,
)

log = logging.getLogger("api")
router = APIRouter()


@router.get("/health")
async def health():
    return {"ok": True, "time": dt.datetime.now(dt.timezone.utc).isoformat()}


def _resolve_mode(raw: Optional[str]) -> SyncMode:
    """
    Приводим строковый режим к SyncMode. Дефолт — settings.default_write_mode.
    """
    val = (raw or settings.default_write_mode or "primary_only").strip().lower()
    try:
        return SyncMode(val)
    except Exception:
        return SyncMode.PRIMARY_ONLY


def _as_int(value: Any, *, default: Optional[int] = None) -> Optional[int]:
    """Безопасно приводит к int или возвращает default."""
    if value is None:
        return default
    try:
        # допускаем числа и строки
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, *, default: Optional[float] = None) -> Optional[float]:
    """Безопасно приводит к float или возвращает default."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@router.post("/v1/analyze/{pars_id}", response_model=AnalyzeResponse)
async def analyze(pars_id: int, body: AnalyzeRequest):
    t0 = dt.datetime.now()
    try:
        # ---- конфигурация моделей ----
        mode: SyncMode = _resolve_mode(getattr(body, "sync_mode", None))
        chat_model: str = (getattr(body, "chat_model", None) or settings.CHAT_MODEL).strip()
        embed_model: str = (getattr(body, "embed_model", None) or settings.embed_model).strip()

        # ---- движки БД ----
        primary_opt: Optional[AsyncEngine] = get_primary_engine()
        secondary_opt: Optional[AsyncEngine] = get_secondary_engine()

        # ---- чтение текста для анализа: предпочитаем primary ----
        read_engine_opt: Optional[AsyncEngine] = primary_opt or secondary_opt
        if read_engine_opt is None:
            raise RuntimeError("Нет доступных подключений к БД (primary/secondary пустые)")

        read_engine: AsyncEngine = cast(AsyncEngine, read_engine_opt)
        read_db: str = "primary" if primary_opt is not None else "secondary"

        text_par_obj: Any = await run_on_engine(read_engine, lambda conn: repo.fetch_text_par(conn, pars_id))
        text_par: str = str(text_par_obj or "")

        # ---- 1) OpenAI → parse ----
        prompt: str = build_prompt(text_par)
        answer: Optional[str] = await call_openai(prompt, chat_model)
        parsed_obj: Any = await parse_openai_answer(answer, text_par, embed_model)
        parsed: Dict[str, Any] = cast(Dict[str, Any], parsed_obj if isinstance(parsed_obj, dict) else {})

        # Готовим строго типизированные значения для prodclass
        prodclass_id_opt: Optional[int] = _as_int(parsed.get("PRODCLASS"))
        prodclass_score_opt: Optional[float] = _as_float(parsed.get("PRODCLASS_SCORE"))

        # ---- 2) enrichment (эмбеддинги + матчинг к справочникам) ----
        goods_catalog_obj: Any = await run_on_engine(
            read_engine, lambda conn: repo.fetch_goods_types_catalog(conn)
        )
        equip_catalog_obj: Any = await run_on_engine(
            read_engine, lambda conn: repo.fetch_equipment_catalog(conn)
        )

        goods_catalog: List[Dict[str, Any]] = cast(List[Dict[str, Any]], goods_catalog_obj or [])
        equip_catalog: List[Dict[str, Any]] = cast(List[Dict[str, Any]], equip_catalog_obj or [])

        goods_source_list: List[str] = cast(List[str], parsed.get("GOODS_TYPE_LIST", []) or [])
        equip_source_list: List[str] = cast(List[str], parsed.get("EQUIPMENT_LIST", []) or [])

        goods_enriched: List[Dict[str, Any]] = await enrich_by_catalog(
            goods_source_list, goods_catalog, embed_model, MATCH_THRESHOLD_GOODS
        )
        equip_enriched: List[Dict[str, Any]] = await enrich_by_catalog(
            equip_source_list, equip_catalog, embed_model, MATCH_THRESHOLD_EQUIPMENT
        )

        # ---- 3) запись результатов (если не dry_run) ----
        report: Dict[str, Any] = {
            "dry_run": bool(getattr(body, "dry_run", False)),
            "read_db": read_db,
            "write_mode": mode.value,
        }

        async def write_all_action(conn) -> Dict[str, Any]:
            added = await repo.ensure_description_column(conn)
            updated = await repo.update_pars_description(conn, pars_id, str(parsed.get("DESCRIPTION", "")))

            # insert_ai_site_prodclass ожидает int и float → жёстко валидируем и даём осмысленную ошибку
            if prodclass_id_opt is None:
                raise ValueError("PRODCLASS отсутствует или не является целым числом")
            if prodclass_score_opt is None:
                raise ValueError("PRODCLASS_SCORE отсутствует или не является числом")

            prodclass_row_id = await repo.insert_ai_site_prodclass(
                conn, pars_id, prodclass_id_opt, float(prodclass_score_opt)
            )

            eq_rows = await repo.insert_ai_site_equipment_enriched(conn, pars_id, equip_enriched)
            gt_rows = await repo.insert_ai_site_goods_types_enriched(conn, pars_id, goods_enriched)

            equip_rows_fmt = [{"id": rid, "equipment": name} for rid, name in (eq_rows or [])]
            goods_rows_fmt = [{"id": rid, "goods_type": name} for rid, name in (gt_rows or [])]
            return {
                "added_description_column": added,
                "updated_pars_site_description": updated,
                "ai_site_prodclass_row_id": prodclass_row_id,
                "prodclass_score": prodclass_score_opt,
                "prodclass_score_source": parsed.get("PRODCLASS_SCORE_SOURCE"),
                "equipment_rows": equip_rows_fmt,
                "goods_type_rows": goods_rows_fmt,
            }

        if not getattr(body, "dry_run", False):
            primary: Optional[AsyncEngine] = primary_opt
            secondary: Optional[AsyncEngine] = secondary_opt

            if mode == SyncMode.PRIMARY_ONLY:
                if primary is None:
                    raise RuntimeError("PRIMARY_ONLY: primary недоступен")
                primary_report_obj: Any = await run_on_engine(primary, write_all_action)
                primary_report: Dict[str, Any] = cast(Dict[str, Any], primary_report_obj)
                report = report | {"write_db": "primary"} | primary_report

            elif mode == SyncMode.DUAL_WRITE:
                if primary is None and secondary is None:
                    raise RuntimeError("DUAL_WRITE: нет доступных БД")

                first_engine_opt: Optional[AsyncEngine] = primary or secondary
                second_engine_opt: Optional[AsyncEngine] = (secondary if first_engine_opt is primary else primary)
                assert first_engine_opt is not None
                first_engine: AsyncEngine = cast(AsyncEngine, first_engine_opt)
                write_db = "primary" if first_engine is primary else "secondary"

                primary_report_obj: Any = await run_on_engine(first_engine, write_all_action)
                primary_report: Dict[str, Any] = cast(Dict[str, Any], primary_report_obj)
                report = report | {"write_db": write_db} | primary_report

                if second_engine_opt is not None:
                    second_engine: AsyncEngine = cast(AsyncEngine, second_engine_opt)
                    try:
                        _ = await run_on_engine(second_engine, write_all_action)
                        report["secondary_mirrored"] = True
                    except Exception as e:
                        log.error("DUAL_WRITE: зеркалирование во вторую БД не удалось: %s", e, exc_info=True)
                        report["secondary_mirrored"] = False
                        report["secondary_error"] = str(e)

            elif mode == SyncMode.FALLBACK_TO_SECONDARY:
                if primary is None and secondary is None:
                    raise RuntimeError("FALLBACK: нет доступных БД")
                try:
                    if primary is None:
                        raise RuntimeError("primary недоступен")
                    primary_report_obj: Any = await run_on_engine(primary, write_all_action)
                    primary_report: Dict[str, Any] = cast(Dict[str, Any], primary_report_obj)
                    report = report | {"write_db": "primary", "fallback_used": False} | primary_report
                except Exception as e:
                    log.error("FALLBACK: ошибка записи в primary, пробуем secondary: %s", e, exc_info=True)
                    if secondary is None:
                        raise
                    secondary_report_obj: Any = await run_on_engine(secondary, write_all_action)
                    secondary_report: Dict[str, Any] = cast(Dict[str, Any], secondary_report_obj)
                    report = report | {
                        "write_db": "secondary",
                        "fallback_used": True,
                        "primary_error": str(e),
                    } | secondary_report

        # ---- ответ ----
        dt_ms = int((dt.datetime.now() - t0).total_seconds() * 1000)

        # Ключ для IB_PRODCLASS.get должен быть int; если prodclass_id_opt отсутствует — используем -1
        prodclass_key: int = prodclass_id_opt if prodclass_id_opt is not None else -1
        prodclass_title: str = IB_PRODCLASS.get(prodclass_key, "—")

        return AnalyzeResponse(
            pars_id=pars_id,
            used_db_mode=report.get("write_mode", ""),
            prompt_len=len(prompt),
            answer_len=len(answer or ""),
            answer_raw=(answer if getattr(body, "return_answer_raw", False) else None),
            parsed=parsed | {"PRODCLASS_TITLE": prodclass_title},
            report=(None if report.get("dry_run") else report),
            duration_ms=dt_ms,
        )
    except Exception as e:
        log.exception("Analyze failed for pars_id=%s: %s", pars_id, e)
        raise HTTPException(status_code=400, detail=str(e))

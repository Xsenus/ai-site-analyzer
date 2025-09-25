# app/api/routes.py
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional, cast

from fastapi import APIRouter, HTTPException
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.exc import ProgrammingError

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
    """Приводим строковый режим к SyncMode. Дефолт — settings.default_write_mode."""
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


# --------- лог-утилиты ---------
def _clip(s: Optional[str], n: int = 200) -> str:
    """Обрезает длинные строки для логов, показывая только начало."""
    if not s:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n] + f"... ({len(s)} chars)"


def _tick(t0: dt.datetime) -> int:
    """Мс с момента t0."""
    return int((dt.datetime.now() - t0).total_seconds() * 1000)


@router.post("/v1/analyze/{pars_id}", response_model=AnalyzeResponse)
async def analyze(pars_id: int, body: AnalyzeRequest):
    t0 = dt.datetime.now()
    log.info("[analyze] start pars_id=%s", pars_id)

    try:
        # ---- Конфигурация моделей ----
        mode: SyncMode = _resolve_mode(getattr(body, "sync_mode", None))
        chat_model_raw = getattr(body, "chat_model", None) or settings.CHAT_MODEL
        embed_model_raw = getattr(body, "embed_model", None) or settings.embed_model
        chat_model: str = (chat_model_raw or "").strip()
        embed_model: str = (embed_model_raw or "").strip()

        log.info("[analyze] config pars_id=%s mode=%s chat_model=%s embed_model=%s",
                 pars_id, mode.value, chat_model or "∅", embed_model or "∅")

        if not chat_model:
            log.error("[analyze] no chat_model pars_id=%s", pars_id)
            raise HTTPException(status_code=400, detail="CHAT_MODEL не задан")
        if not embed_model:
            log.error("[analyze] no embed_model pars_id=%s", pars_id)
            raise HTTPException(status_code=400, detail="embed_model не задан")

        # ---- Движки БД ----
        primary_opt: Optional[AsyncEngine] = get_primary_engine()
        secondary_opt: Optional[AsyncEngine] = get_secondary_engine()
        log.info("[analyze] engines pars_id=%s primary=%s secondary=%s",
                 pars_id, "yes" if primary_opt else "no", "yes" if secondary_opt else "no")

        # ---- Чтение текста для анализа: предпочитаем primary ----
        read_engine_opt: Optional[AsyncEngine] = primary_opt or secondary_opt
        if read_engine_opt is None:
            log.error("[analyze] no DB engines available pars_id=%s", pars_id)
            raise HTTPException(status_code=500, detail="Нет доступных подключений к БД (primary/secondary пустые)")

        read_engine: AsyncEngine = cast(AsyncEngine, read_engine_opt)
        read_db: str = "primary" if primary_opt is not None else "secondary"
        log.info("[analyze] read_db=%s pars_id=%s", read_db, pars_id)

        # чтение текста
        t_read = dt.datetime.now()
        text_par_obj: Any = await run_on_engine(read_engine, lambda conn: repo.fetch_text_par(conn, pars_id))
        ms_read = _tick(t_read)
        text_par: str = str(text_par_obj or "").strip()
        log.debug("[analyze] fetched text pars_id=%s len=%s sample=%r took_ms=%s",
                  pars_id, len(text_par), _clip(text_par), ms_read)

        if not text_par:
            log.error("[analyze] empty source text pars_id=%s", pars_id)
            raise HTTPException(status_code=400, detail=f"Нет исходного текста для pars_id={pars_id}")

        # ---- 1) OpenAI → parse ----
        # build prompt
        t_prompt = dt.datetime.now()
        prompt: str = build_prompt(text_par)
        ms_prompt = _tick(t_prompt)
        log.debug("[analyze] prompt built pars_id=%s len=%s sample=%r took_ms=%s",
                  pars_id, len(prompt), _clip(prompt), ms_prompt)

        # call LLM
        t_llm = dt.datetime.now()
        answer: Optional[str] = await call_openai(prompt, chat_model)
        ms_llm = _tick(t_llm)
        log.info("[analyze] LLM called pars_id=%s chat_model=%s took_ms=%s ok=%s",
                 pars_id, chat_model, ms_llm, bool(answer))
        log.debug("[analyze] LLM answer pars_id=%s len=%s sample=%r",
                  pars_id, len(answer or ""), _clip(answer))

        if answer is None or not str(answer).strip():
            log.error("[analyze] empty LLM answer pars_id=%s", pars_id)
            raise HTTPException(status_code=502, detail="LLM вернул пустой ответ")

        # parse LLM answer
        t_parse = dt.datetime.now()
        parsed_obj: Any = await parse_openai_answer(answer, text_par, embed_model)
        ms_parse = _tick(t_parse)
        if not isinstance(parsed_obj, dict):
            log.error("[analyze] parser returned non-dict pars_id=%s type=%s took_ms=%s",
                      pars_id, type(parsed_obj).__name__, ms_parse)
            raise HTTPException(status_code=502, detail="Парсер ответа LLM вернул некорректную структуру")

        parsed: Dict[str, Any] = cast(Dict[str, Any], parsed_obj)
        log.info("[analyze] parsed done pars_id=%s took_ms=%s keys=%s",
                 pars_id, ms_parse, sorted(parsed.keys()))
        log.debug("[analyze] parsed snippet pars_id=%s %r", pars_id,
                  {k: parsed.get(k) for k in ("DESCRIPTION", "PRODCLASS", "PRODCLASS_SCORE",
                                              "EQUIPMENT_LIST", "GOODS_TYPE_LIST")})

        # Строго типизированные значения для prodclass
        prodclass_id_opt: Optional[int] = _as_int(parsed.get("PRODCLASS"))
        prodclass_score_opt: Optional[float] = _as_float(parsed.get("PRODCLASS_SCORE"))
        log.info("[analyze] prodclass parsed pars_id=%s id=%s score=%s",
                 pars_id, prodclass_id_opt, prodclass_score_opt)

        # ---- 2) enrichment (эмбеддинги + матчинг к справочникам) ----
        t_cat = dt.datetime.now()
        goods_catalog_obj: Any = await run_on_engine(read_engine, lambda conn: repo.fetch_goods_types_catalog(conn))
        equip_catalog_obj: Any = await run_on_engine(read_engine, lambda conn: repo.fetch_equipment_catalog(conn))
        ms_cat = _tick(t_cat)
        goods_catalog: List[Dict[str, Any]] = cast(List[Dict[str, Any]], goods_catalog_obj or [])
        equip_catalog: List[Dict[str, Any]] = cast(List[Dict[str, Any]], equip_catalog_obj or [])
        log.info("[analyze] catalogs loaded pars_id=%s goods=%s equip=%s took_ms=%s",
                 pars_id, len(goods_catalog), len(equip_catalog), ms_cat)

        goods_source_list: List[str] = cast(List[str], parsed.get("GOODS_TYPE_LIST", []) or [])
        equip_source_list: List[str] = cast(List[str], parsed.get("EQUIPMENT_LIST", []) or [])
        log.info("[analyze] sources pars_id=%s goods_src=%s equip_src=%s",
                 pars_id, len(goods_source_list), len(equip_source_list))
        log.debug("[analyze] sources samples pars_id=%s goods=%r equip=%r",
                  pars_id, goods_source_list[:5], equip_source_list[:5])

        # обогащение
        t_enrich = dt.datetime.now()
        goods_enriched: List[Dict[str, Any]] = await enrich_by_catalog(
            goods_source_list, goods_catalog, embed_model, MATCH_THRESHOLD_GOODS
        )
        equip_enriched: List[Dict[str, Any]] = await enrich_by_catalog(
            equip_source_list, equip_catalog, embed_model, MATCH_THRESHOLD_EQUIPMENT
        )
        ms_enrich = _tick(t_enrich)
        log.info("[analyze] enrichment done pars_id=%s took_ms=%s goods_enriched=%s equip_enriched=%s",
                 pars_id, ms_enrich, len(goods_enriched), len(equip_enriched))
        log.debug("[analyze] goods_enriched sample pars_id=%s %r", pars_id, goods_enriched[:3])
        log.debug("[analyze] equip_enriched sample pars_id=%s %r", pars_id, equip_enriched[:3])

        # ---- 3) запись результатов (если не dry_run) ----
        is_dry = bool(getattr(body, "dry_run", False))
        report: Dict[str, Any] = {
            "dry_run": is_dry,
            "read_db": read_db,
            "write_mode": mode.value,
            "counts": {
                "goods_source": len(goods_source_list),
                "equip_source": len(equip_source_list),
                "goods_enriched": len(goods_enriched),
                "equip_enriched": len(equip_enriched),
            },
        }
        log.info("[analyze] write phase pars_id=%s dry_run=%s mode=%s", pars_id, is_dry, mode.value)

        async def write_all_action(conn) -> Dict[str, Any]:
            t_w = dt.datetime.now()
            log.info("[analyze/write] begin pars_id=%s", pars_id)

            added = await repo.ensure_description_column(conn)
            log.info("[analyze/write] ensure_description_column pars_id=%s added=%s", pars_id, added)

            updated = await repo.update_pars_description(conn, pars_id, str(parsed.get("DESCRIPTION", "")))
            log.info("[analyze/write] update_pars_description pars_id=%s updated=%s", pars_id, updated)

            # Жёсткая валидация входа для prodclass
            if prodclass_id_opt is None:
                log.error("[analyze/write] prodclass_id missing pars_id=%s", pars_id)
                raise ValueError("PRODCLASS отсутствует или не является целым числом")
            if prodclass_score_opt is None:
                log.error("[analyze/write] prodclass_score missing pars_id=%s", pars_id)
                raise ValueError("PRODCLASS_SCORE отсутствует или не является числом")

            prodclass_row_id = await repo.insert_ai_site_prodclass(
                conn, pars_id, prodclass_id_opt, float(prodclass_score_opt)
            )
            log.info("[analyze/write] insert_ai_site_prodclass pars_id=%s row_id=%s id=%s score=%s",
                     pars_id, prodclass_row_id, prodclass_id_opt, prodclass_score_opt)

            eq_rows = await repo.insert_ai_site_equipment_enriched(conn, pars_id, equip_enriched)
            log.info("[analyze/write] insert_ai_site_equipment_enriched pars_id=%s inserted=%s",
                     pars_id, len(eq_rows))

            gt_rows = await repo.insert_ai_site_goods_types_enriched(conn, pars_id, goods_enriched)
            log.info("[analyze/write] insert_ai_site_goods_types_enriched pars_id=%s inserted=%s",
                     pars_id, len(gt_rows))

            equip_rows_fmt = [{"id": rid, "equipment": name} for rid, name in (eq_rows or [])]
            goods_rows_fmt = [{"id": rid, "goods_type": name} for rid, name in (gt_rows or [])]

            ms_write = _tick(t_w)
            log.info("[analyze/write] done pars_id=%s took_ms=%s", pars_id, ms_write)

            return {
                "added_description_column": added,
                "updated_pars_site_description": updated,
                "ai_site_prodclass_row_id": prodclass_row_id,
                "prodclass_score": prodclass_score_opt,
                "prodclass_score_source": parsed.get("PRODCLASS_SCORE_SOURCE"),
                "equipment_rows": equip_rows_fmt,
                "goods_type_rows": goods_rows_fmt,
            }

        if not is_dry:
            primary: Optional[AsyncEngine] = primary_opt
            secondary: Optional[AsyncEngine] = secondary_opt

            try:
                if mode == SyncMode.PRIMARY_ONLY:
                    if primary is None:
                        log.error("[analyze/write] PRIMARY_ONLY but primary is None pars_id=%s", pars_id)
                        raise RuntimeError("PRIMARY_ONLY: primary недоступен")
                    t_db = dt.datetime.now()
                    primary_report_obj: Any = await run_on_engine(primary, write_all_action)
                    ms_db = _tick(t_db)
                    primary_report: Dict[str, Any] = cast(Dict[str, Any], primary_report_obj)
                    report = report | {"write_db": "primary"} | primary_report
                    log.info("[analyze/write] PRIMARY_ONLY ok pars_id=%s took_ms=%s", pars_id, ms_db)

                elif mode == SyncMode.DUAL_WRITE:
                    if primary is None and secondary is None:
                        log.error("[analyze/write] DUAL_WRITE but no engines pars_id=%s", pars_id)
                        raise RuntimeError("DUAL_WRITE: нет доступных БД")
                    first_engine_opt: Optional[AsyncEngine] = primary or secondary
                    second_engine_opt: Optional[AsyncEngine] = (secondary if first_engine_opt is primary else primary)
                    assert first_engine_opt is not None
                    first_engine: AsyncEngine = cast(AsyncEngine, first_engine_opt)
                    write_db = "primary" if first_engine is primary else "secondary"
                    log.info("[analyze/write] DUAL_WRITE first=%s second=%s pars_id=%s",
                             write_db, ("secondary" if write_db == "primary" else "primary"), pars_id)

                    t_db1 = dt.datetime.now()
                    primary_report_obj: Any = await run_on_engine(first_engine, write_all_action)
                    ms_db1 = _tick(t_db1)
                    primary_report: Dict[str, Any] = cast(Dict[str, Any], primary_report_obj)
                    report = report | {"write_db": write_db} | primary_report
                    log.info("[analyze/write] DUAL_WRITE first ok pars_id=%s took_ms=%s", pars_id, ms_db1)

                    if second_engine_opt is not None:
                        second_engine: AsyncEngine = cast(AsyncEngine, second_engine_opt)
                        try:
                            t_db2 = dt.datetime.now()
                            _ = await run_on_engine(second_engine, write_all_action)
                            ms_db2 = _tick(t_db2)
                            report["secondary_mirrored"] = True
                            log.info("[analyze/write] DUAL_WRITE mirror ok pars_id=%s took_ms=%s", pars_id, ms_db2)
                        except Exception as e:
                            log.error("DUAL_WRITE: зеркалирование во вторую БД не удалось pars_id=%s: %s",
                                      pars_id, e, exc_info=True)
                            report["secondary_mirrored"] = False
                            report["secondary_error"] = str(e)

                elif mode == SyncMode.FALLBACK_TO_SECONDARY:
                    if primary is None and secondary is None:
                        log.error("[analyze/write] FALLBACK no engines pars_id=%s", pars_id)
                        raise RuntimeError("FALLBACK: нет доступных БД")
                    try:
                        if primary is None:
                            raise RuntimeError("primary недоступен")
                        t_db = dt.datetime.now()
                        primary_report_obj: Any = await run_on_engine(primary, write_all_action)
                        ms_db = _tick(t_db)
                        primary_report: Dict[str, Any] = cast(Dict[str, Any], primary_report_obj)
                        report = report | {"write_db": "primary", "fallback_used": False} | primary_report
                        log.info("[analyze/write] FALLBACK primary ok pars_id=%s took_ms=%s", pars_id, ms_db)
                    except Exception as e:
                        log.error("FALLBACK: ошибка записи в primary pars_id=%s, пробуем secondary: %s",
                                  pars_id, e, exc_info=True)
                        if secondary is None:
                            raise
                        t_db2 = dt.datetime.now()
                        secondary_report_obj: Any = await run_on_engine(secondary, write_all_action)
                        ms_db2 = _tick(t_db2)
                        secondary_report: Dict[str, Any] = cast(Dict[str, Any], secondary_report_obj)
                        report = report | {
                            "write_db": "secondary",
                            "fallback_used": True,
                            "primary_error": str(e),
                        } | secondary_report
                        log.info("[analyze/write] FALLBACK secondary ok pars_id=%s took_ms=%s", pars_id, ms_db2)

            except ProgrammingError as pe:
                # Явно перехватываем кейс со смешанными плейсхолдерами (:vec vs %(name)s)
                msg = str(pe)
                if ":vec" in msg or ' at or near ":"' in msg:
                    log.error("[analyze/write] vector placeholder mismatch pars_id=%s msg=%s", pars_id, msg)
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "SQL ошибка параметра vector: используйте единый стиль плейсхолдеров. "
                            "Замените ':vec::vector' на CAST(:vec AS vector) или привязку через тип."
                        ),
                    )
                log.error("[analyze/write] ProgrammingError pars_id=%s: %s", pars_id, msg, exc_info=True)
                raise

        # ---- Ответ ----
        dt_ms = _tick(t0)
        prodclass_key: int = prodclass_id_opt if prodclass_id_opt is not None else -1
        prodclass_title: str = IB_PRODCLASS.get(prodclass_key, "—")

        log.info("[analyze] done pars_id=%s duration_ms=%s used_db_mode=%s write_db=%s",
                 pars_id, dt_ms, report.get("write_mode", ""), report.get("write_db", "∅"))

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

    except HTTPException:
        log.info("[analyze] http-exception pars_id=%s duration_ms=%s", pars_id, _tick(t0))
        raise
    except Exception as e:
        log.exception("Analyze failed for pars_id=%s: %s", pars_id, e)
        raise HTTPException(status_code=400, detail=str(e))

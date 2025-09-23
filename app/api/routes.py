from __future__ import annotations
import datetime as dt
import logging
from fastapi import APIRouter, HTTPException

from app.config import settings
from app.db.tx import dual_write, SyncMode
from app.api.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.analyzer import (
    build_prompt, call_openai, parse_openai_answer,
    enrich_by_catalog, MATCH_THRESHOLD_EQUIPMENT, MATCH_THRESHOLD_GOODS,
)
from app.models.ib_prodclass import IB_PRODCLASS
from app.repositories import parsing_repo as repo

log = logging.getLogger("api")
router = APIRouter()


@router.get("/health")
async def health():
    return {"ok": True, "time": dt.datetime.now(dt.timezone.utc).isoformat()}


@router.post("/v1/analyze/{pars_id}", response_model=AnalyzeResponse)
async def analyze(pars_id: int, body: AnalyzeRequest):
    t0 = dt.datetime.now()
    try:
        mode: SyncMode = (body.sync_mode or settings.DEFAULT_SYNC_MODE)
        chat_model = body.chat_model or settings.CHAT_MODEL
        embed_model = body.embed_model or settings.EMBED_MODEL

        async with dual_write(mode) as db:
            used = db["used"]
            primary = db.get("primary")
            secondary = db.get("secondary")

            read_conn = primary or secondary
            text_par = await repo.fetch_text_par(read_conn, pars_id)

            # 1) OpenAI → parse
            prompt = build_prompt(text_par)
            answer = await call_openai(prompt, chat_model)
            parsed = await parse_openai_answer(answer, text_par, embed_model)

            # 2) Enrichment (эмбеддинги + матчинг к справочникам)
            goods_catalog = await repo.fetch_goods_types_catalog(read_conn)
            equip_catalog = await repo.fetch_equipment_catalog(read_conn)

            goods_enriched = await enrich_by_catalog(
                parsed["GOODS_TYPE_LIST"], goods_catalog, embed_model, MATCH_THRESHOLD_GOODS
            )
            equip_enriched = await enrich_by_catalog(
                parsed["EQUIPMENT_LIST"], equip_catalog, embed_model, MATCH_THRESHOLD_EQUIPMENT
            )

            report = {"dry_run": True, "used": used}
            if not body.dry_run:
                async def write_all(conn):
                    added = await repo.ensure_description_column(conn)
                    updated = await repo.update_pars_description(conn, pars_id, parsed["DESCRIPTION"])
                    prodclass_id = await repo.insert_ai_site_prodclass(conn, pars_id, parsed["PRODCLASS"], parsed["PRODCLASS_SCORE"])
                    eq_rows = await repo.insert_ai_site_equipment_enriched(conn, pars_id, equip_enriched)
                    gt_rows = await repo.insert_ai_site_goods_types_enriched(conn, pars_id, goods_enriched)
                    return {
                        "added_description_column": added,
                        "updated_pars_site_description": updated,
                        "ai_site_prodclass_row_id": prodclass_id,
                        "prodclass_score": parsed["PRODCLASS_SCORE"],
                        "prodclass_score_source": parsed["PRODCLASS_SCORE_SOURCE"],
                        "equipment_rows": [{"id": rid, "equipment": name} for rid, name in eq_rows],
                        "goods_type_rows": [{"id": rid, "goods_type": name} for rid, name in gt_rows],
                    }

                primary_report = await write_all(read_conn)

                if secondary and (read_conn is primary):
                    try:
                        await write_all(secondary)
                        primary_report["secondary_mirrored"] = True
                    except Exception as e:
                        log.error("Secondary mirror failed: %s", e)
                        primary_report["secondary_mirrored"] = False
                        primary_report["secondary_error"] = str(e)

                report = primary_report | {"used": used}

        dt_ms = int((dt.datetime.now() - t0).total_seconds() * 1000)
        return AnalyzeResponse(
            pars_id=pars_id,
            used_db_mode=str(report.get("used", "")),
            prompt_len=len(prompt),
            answer_len=len(answer or ""),
            answer_raw=(answer if body.return_answer_raw else None),
            parsed=parsed | {"PRODCLASS_TITLE": IB_PRODCLASS.get(parsed["PRODCLASS"], "—")},
            report=(report if not body.dry_run else None),
            duration_ms=dt_ms,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

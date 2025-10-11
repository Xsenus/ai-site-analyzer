from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, cast
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Body, Query
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.exc import ProgrammingError

from app.config import settings
from app.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalyzeFromJsonRequest,
    AnalyzeFromJsonResponse,
    CatalogItem,
    DbPayload,
    IbMatchItem,
    IbMatchRequest,
    IbMatchResponse,
    IbMatchSummary,
    MatchedItemPayload,
    ParsSiteInfo,
    ProdclassPayload,
    SitePipelineIbMatchOptions,
    SitePipelineParseResult,
    SitePipelineRequest,
    SitePipelineResolved,
    SitePipelineResponse,
    VectorPayload,
)
from app.schemas.equipment_selection import (
    ClientRow,
    EquipmentSelectionResponse,
    SampleTable,
)
from app.services.analyzer import (
    build_prompt,
    call_openai,
    parse_openai_answer,
    enrich_by_catalog,
    MATCH_THRESHOLD_EQUIPMENT,
    MATCH_THRESHOLD_GOODS,
    embed_single_text,
)
from app.models.ib_prodclass import IB_PRODCLASS
from app.repositories import parsing_repo as repo
from app.db.tx import (
    SyncMode,
    get_primary_engine,
    get_secondary_engine,
    run_on_engine,
    dual_write,
)
from app.db.parsing import ping_parsing
from app.db.postgres import ping_postgres
from app.services.embeddings import embed_many
from app.services.equipment_selection import compute_equipment_selection
from app.utils.vectors import cosine_similarity, format_pgvector, parse_pgvector

log = logging.getLogger("api")
router = APIRouter()


def _render_ascii_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    aligns: list[str] | None = None,
) -> list[str]:
    """Простая ASCII-таблица для включения в текстовый отчёт."""

    if not rows:
        widths = [len(h) for h in headers]
    else:
        widths = [len(h) for h in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], len(cell))

    def _format_cell(value: str, width: int, align: str) -> str:
        if align == "right":
            return value.rjust(width)
        if align == "center":
            return value.center(width)
        return value.ljust(width)

    aligns = (aligns or ["left"] * len(headers))[: len(headers)]
    if len(aligns) < len(headers):
        aligns = aligns + ["left"] * (len(headers) - len(aligns))
    top_border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    header_line = (
        "| "
        + " | ".join(
            _format_cell(str(h), widths[idx], aligns[idx]) for idx, h in enumerate(headers)
        )
        + " |"
    )
    header_sep = (
        "|"
        + "+".join("-" * (w + 2) for w in widths)
        + "|"
    )

    lines = [top_border, header_line, header_sep]
    for row in rows:
        formatted = (
            "| "
            + " | ".join(
                _format_cell(cell, widths[idx], aligns[idx]) for idx, cell in enumerate(row)
            )
            + " |"
        )
        lines.append(formatted)
    lines.append(top_border)
    return lines


@router.get("/health")
async def health() -> Dict[str, Any]:
    """Расширенный health-check, пингующий обе БД."""

    try:
        parsing_ok = await ping_parsing()
    except Exception as exc:  # pragma: no cover - защитный лог
        log.error("Health ping_parsing failed: %s", exc)
        parsing_ok = False

    try:
        postgres_ok = await ping_postgres()
    except Exception as exc:  # pragma: no cover - защитный лог
        log.error("Health ping_postgres failed: %s", exc)
        postgres_ok = False

    connections = {
        "parsing_data": parsing_ok,
        "postgres": postgres_ok,
    }
    ok = all(connections.values()) if connections else False

    return {
        "ok": ok,
        "time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "connections": connections,
    }


@router.get("/v1/equipment-selection", response_model=EquipmentSelectionResponse)
async def equipment_selection(
    client_request_id: int = Query(..., ge=1)
) -> EquipmentSelectionResponse:
    """
    Вычисляет таблицы оборудования (1way/2way/3way/all) для клиента и возвращает
    срезы данных, чтобы сразу увидеть результат.
    """
    engine_opt: Optional[AsyncEngine] = get_primary_engine() or get_secondary_engine()
    if engine_opt is None:
        raise HTTPException(status_code=500, detail="Нет доступных подключений к БД")

    log.info("[equipment-selection] start client_request_id=%s", client_request_id)

    async def _action(conn):
        return await compute_equipment_selection(conn, client_request_id)

    result = await run_on_engine(engine_opt, _action)

    # --- Собираем наглядные ASCII-таблицы по топ-5 записей из каждого списка ---
    sample_tables: list[SampleTable] = []

    def _append_sample(title: str, headers: list[str], rows: list[list[str]]) -> None:
        if not rows:
            return
        sample_tables.append(
            SampleTable(title=title, lines=_render_ascii_table(headers, rows))
        )

    def _format_score(value: float | None) -> str:
        return "-" if value is None else f"{value:.4f}"

    _append_sample(
        "EQUIPMENT_1way (через prodclass)",
        ["ID", "Название", "SCORE"],
        [
            [str(row.id), row.equipment_name or "-", _format_score(row.score)]
            for row in result.equipment_1way[:5]
        ],
    )
    _append_sample(
        "EQUIPMENT_2way (через goods_type)",
        ["ID", "Название", "SCORE"],
        [
            [str(row.id), row.equipment_name or "-", _format_score(row.score)]
            for row in result.equipment_2way[:5]
        ],
    )
    _append_sample(
        "EQUIPMENT_3way (через ai_site_equipment)",
        ["ID", "Название", "SCORE"],
        [
            [str(row.id), row.equipment_name or "-", _format_score(row.score)]
            for row in result.equipment_3way[:5]
        ],
    )
    _append_sample(
        "EQUIPMENT_ALL (объединённый рейтинг)",
        ["ID", "Название", "SCORE", "Источник"],
        [
            [
                str(row.id),
                row.equipment_name or "-", 
                _format_score(row.score),
                row.source,
            ]
            for row in result.equipment_all[:5]
        ],
    )

    result = result.copy(update={"sample_tables": sample_tables})
    log.info(
        "[equipment-selection] done client_request_id=%s equipment_all=%s",
        client_request_id,
        len(result.equipment_all),
    )
    return result


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


def _catalog_items_to_dict(items: Optional[List[CatalogItem]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not items:
        return normalized

    for item in items:
        vec = item.vec
        vec_str: Optional[str] = None
        if isinstance(vec, list):
            try:
                floats = [float(v) for v in vec]
            except (TypeError, ValueError):
                log.warning(
                    "[analyze/json] catalog vector conversion failed id=%s name=%r", item.id, item.name
                )
                floats = []
            vec_str = format_pgvector(floats) if floats else "[]"
        elif isinstance(vec, str):
            vec_str = vec.strip() or None
        else:
            vec_str = None

        normalized.append({"id": item.id, "name": item.name, "vec": vec_str})

    return normalized


def _vector_payload(values: Optional[List[float]], literal: Optional[str]) -> VectorPayload:
    safe_values: Optional[List[float]] = None
    if values:
        try:
            safe_values = [float(v) for v in values]
        except (TypeError, ValueError):
            log.warning("[analyze/json] failed to normalize vector values length=%s", len(values))
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
        vector = _vector_payload(vec_values if isinstance(vec_values, list) else None, vec_literal)
        payload.append(
            MatchedItemPayload(
                text=str(it.get("text") or ""),
                match_id=_as_int(it.get("match_id")),
                score=_as_float(it.get("score")),
                vector=vector,
            )
        )
    return payload


def _normalize_site(raw: str) -> str:
    """
    Принимает домен или полный URL и приводит к нормализованному домену:
    - режет схему/путь/квери/фрагмент/порт
    - убирает префикс www.
    - срезает хвостовые точки и слэши
    - переводит в IDNA (punycode) при необходимости
    """
    s = (raw or "").strip().lower()
    if not s:
        return ""

    # Если выглядит как "host/path" без схемы — добавим фальш-схему, чтобы корректно распарсить
    if "://" not in s and ("/" in s or "?" in s or "#" in s):
        s = "http://" + s

    host: str
    if "://" in s:
        p = urlparse(s)
        host = p.hostname or ""
    else:
        # просто домен, возможно с портом
        host = s.split("/", 1)[0]

    # Уберём порт, www., обрежем мусор
    host = host.split(":", 1)[0]
    host = re.sub(r"^www\.", "", host).strip().strip("/").rstrip(".")
    if not host:
        return ""

    # IDNA
    try:
        host = host.encode("idna").decode("ascii")
    except Exception:
        pass

    return host


# ======================
# Присвоение справочников IB
# ======================
@router.post("/v1/ib-match", response_model=IbMatchResponse)
async def assign_ib_matches(body: IbMatchRequest) -> IbMatchResponse:
    started = dt.datetime.now()
    client_id = body.client_id
    mode = _resolve_mode(getattr(body, "sync_mode", None))

    log.info(
        "[ib-match] start client_id=%s mode=%s reembed=%s",
        client_id,
        mode.value,
        body.reembed_if_exists,
    )

    primary = get_primary_engine()
    secondary = get_secondary_engine()
    read_engine = primary or secondary
    if read_engine is None:
        log.error("[ib-match] no DB engines available client_id=%s", client_id)
        raise HTTPException(
            status_code=500,
            detail="Нет доступных подключений к БД (primary/secondary пустые)",
        )

    try:
        goods_rows: List[dict] = await run_on_engine(
            read_engine, lambda conn: repo.fetch_client_goods(conn, client_id)
        )
        equip_rows: List[dict] = await run_on_engine(
            read_engine, lambda conn: repo.fetch_client_equipment(conn, client_id)
        )
        goods_catalog = await run_on_engine(read_engine, lambda conn: repo.fetch_goods_types_catalog(conn))
        equip_catalog = await run_on_engine(read_engine, lambda conn: repo.fetch_equipment_catalog(conn))
    except Exception as exc:
        log.error("[ib-match] DB read error client_id=%s: %s", client_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка чтения БД: {exc}")

    log.info(
        "[ib-match] fetched client_id=%s goods=%s equipment=%s catalog_goods=%s catalog_equipment=%s",
        client_id,
        len(goods_rows),
        len(equip_rows),
        len(goods_catalog),
        len(equip_catalog),
    )

    goods_catalog_vecs = [
        (item["id"], item["name"], parse_pgvector(item.get("vec"))) for item in goods_catalog
    ]
    goods_catalog_vecs = [
        (gid, name, vec)
        for gid, name, vec in goods_catalog_vecs
        if isinstance(vec, list) and vec
    ]

    equip_catalog_vecs = [
        (item["id"], item["name"], parse_pgvector(item.get("vec"))) for item in equip_catalog
    ]
    equip_catalog_vecs = [
        (eid, name, vec)
        for eid, name, vec in equip_catalog_vecs
        if isinstance(vec, list) and vec
    ]

    goods_map = {row["id"]: row for row in goods_rows}
    equip_map = {row["id"]: row for row in equip_rows}

    goods_to_embed = []
    for row in goods_rows:
        vec = parse_pgvector(row.get("vec"))
        if body.reembed_if_exists or not vec:
            text = (row.get("text") or "").strip()
            if text:
                goods_to_embed.append((row["id"], text))

    equip_to_embed = []
    for row in equip_rows:
        vec = parse_pgvector(row.get("vec"))
        if body.reembed_if_exists or not vec:
            text = (row.get("text") or "").strip()
            if text:
                equip_to_embed.append((row["id"], text))

    goods_embedded = 0
    equipment_embedded = 0

    if goods_to_embed:
        texts = [text for _, text in goods_to_embed]
        log.info("[ib-match] embedding goods items=%s", len(texts))
        try:
            vectors = await embed_many(texts, timeout=settings.AI_SEARCH_TIMEOUT or 12.0)
        except Exception as exc:
            log.error("[ib-match] goods embedding failed client_id=%s: %s", client_id, exc, exc_info=True)
            raise HTTPException(status_code=502, detail="Не удалось получить эмбеддинги для goods_types")

        goods_updates: List[tuple[int, str]] = []
        for (gid, _), vec in zip(goods_to_embed, vectors):
            if vec:
                vec_str = format_pgvector(vec)
                goods_updates.append((gid, vec_str))
                goods_map[gid]["vec"] = vec_str
        goods_embedded = len(goods_updates)

        if goods_updates:
            try:
                await dual_write(lambda conn: repo.update_goods_vectors(conn, goods_updates), mode=mode)
            except Exception as exc:
                log.error("[ib-match] update goods vectors failed client_id=%s: %s", client_id, exc, exc_info=True)
                raise HTTPException(status_code=500, detail=f"Ошибка сохранения эмбеддингов goods_types: {exc}")

    if equip_to_embed:
        texts = [text for _, text in equip_to_embed]
        log.info("[ib-match] embedding equipment items=%s", len(texts))
        try:
            vectors = await embed_many(texts, timeout=settings.AI_SEARCH_TIMEOUT or 12.0)
        except Exception as exc:
            log.error("[ib-match] equipment embedding failed client_id=%s: %s", client_id, exc, exc_info=True)
            raise HTTPException(status_code=502, detail="Не удалось получить эмбеддинги для equipment")

        equip_updates: List[tuple[int, str]] = []
        for (eid, _), vec in zip(equip_to_embed, vectors):
            if vec:
                vec_str = format_pgvector(vec)
                equip_updates.append((eid, vec_str))
                equip_map[eid]["vec"] = vec_str
        equipment_embedded = len(equip_updates)

        if equip_updates:
            try:
                await dual_write(lambda conn: repo.update_equipment_vectors(conn, equip_updates), mode=mode)
            except Exception as exc:
                log.error("[ib-match] update equipment vectors failed client_id=%s: %s", client_id, exc, exc_info=True)
                raise HTTPException(status_code=500, detail=f"Ошибка сохранения эмбеддингов equipment: {exc}")

    goods_matches: List[IbMatchItem] = []
    goods_updates_db: List[tuple[int, int, float]] = []

    for row in goods_rows:
        vec = parse_pgvector(row.get("vec"))
        if not vec:
            goods_matches.append(
                IbMatchItem(
                    ai_id=row["id"],
                    source_text=row.get("text") or "",
                    match_id=None,
                    match_name=None,
                    score=None,
                    note="Нет вектора — пропуск",
                )
            )
            continue

        best_id: Optional[int] = None
        best_name: Optional[str] = None
        best_score = -1.0

        for ib_id, ib_name, ib_vec in goods_catalog_vecs:
            score = cosine_similarity(vec, ib_vec)
            if score > best_score:
                best_id = ib_id
                best_name = ib_name
                best_score = score

        if best_id is None:
            goods_matches.append(
                IbMatchItem(
                    ai_id=row["id"],
                    source_text=row.get("text") or "",
                    match_id=None,
                    match_name=None,
                    score=None,
                    note="Нет кандидатов в справочнике",
                )
            )
            continue

        goods_matches.append(
            IbMatchItem(
                ai_id=row["id"],
                source_text=row.get("text") or "",
                match_id=best_id,
                match_name=best_name,
                score=round(best_score, 4),
                note=None,
            )
        )
        goods_updates_db.append((row["id"], best_id, round(best_score, 2)))

    equipment_matches: List[IbMatchItem] = []
    equip_updates_db: List[tuple[int, int, float]] = []

    for row in equip_rows:
        vec = parse_pgvector(row.get("vec"))
        if not vec:
            equipment_matches.append(
                IbMatchItem(
                    ai_id=row["id"],
                    source_text=row.get("text") or "",
                    match_id=None,
                    match_name=None,
                    score=None,
                    note="Нет вектора — пропуск",
                )
            )
            continue

        best_id: Optional[int] = None
        best_name: Optional[str] = None
        best_score = -1.0

        for ib_id, ib_name, ib_vec in equip_catalog_vecs:
            score = cosine_similarity(vec, ib_vec)
            if score > best_score:
                best_id = ib_id
                best_name = ib_name
                best_score = score

        if best_id is None:
            equipment_matches.append(
                IbMatchItem(
                    ai_id=row["id"],
                    source_text=row.get("text") or "",
                    match_id=None,
                    match_name=None,
                    score=None,
                    note="Нет кандидатов в справочнике",
                )
            )
            continue

        equipment_matches.append(
            IbMatchItem(
                ai_id=row["id"],
                source_text=row.get("text") or "",
                match_id=best_id,
                match_name=best_name,
                score=round(best_score, 4),
                note=None,
            )
        )
        equip_updates_db.append((row["id"], best_id, round(best_score, 2)))

    if goods_updates_db:
        try:
            await dual_write(lambda conn: repo.update_goods_matches(conn, goods_updates_db), mode=mode)
        except Exception as exc:
            log.error("[ib-match] update goods matches failed client_id=%s: %s", client_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Ошибка сохранения соответствий goods_types: {exc}")

    if equip_updates_db:
        try:
            await dual_write(lambda conn: repo.update_equipment_matches(conn, equip_updates_db), mode=mode)
        except Exception as exc:
            log.error("[ib-match] update equipment matches failed client_id=%s: %s", client_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Ошибка сохранения соответствий equipment: {exc}")

    for item in goods_matches:
        if item.match_id is not None and item.score is not None:
            log.info(
                "[ib-match] goods id=%s '%s' -> ib_id=%s '%s' score=%.4f",
                item.ai_id,
                _clip(item.source_text),
                item.match_id,
                _clip(item.match_name or ""),
                item.score,
            )

    for item in equipment_matches:
        if item.match_id is not None and item.score is not None:
            log.info(
                "[ib-match] equipment id=%s '%s' -> ib_id=%s '%s' score=%.4f",
                item.ai_id,
                _clip(item.source_text),
                item.match_id,
                _clip(item.match_name or ""),
                item.score,
            )

    summary = IbMatchSummary(
        goods_total=len(goods_rows),
        goods_updated=len(goods_updates_db),
        goods_embedded=goods_embedded,
        equipment_total=len(equip_rows),
        equipment_updated=len(equip_updates_db),
        equipment_embedded=equipment_embedded,
        catalog_goods_total=len(goods_catalog_vecs),
        catalog_equipment_total=len(equip_catalog_vecs),
    )

    separator = "=" * 88
    debug_lines: list[str] = []
    debug_lines.append(
        f"[INFO] Найдено записей по CLIENT_ID={client_id}: "
        f"goods_types={len(goods_rows)}, equipment={len(equip_rows)}"
    )

    if goods_rows:
        debug_lines.append("")
        debug_lines.append("— Связанные ai_site_goods_types.goods_type:")
        for row in goods_rows:
            debug_lines.append(f"   [goods_id={row['id']}] {row.get('text', '')}")

    if equip_rows:
        debug_lines.append("")
        debug_lines.append("— Связанные ai_site_equipment.equipment:")
        for row in equip_rows:
            debug_lines.append(f"   [equip_id={row['id']}] {row.get('text', '')}")

    debug_lines.append("")
    debug_lines.append(
        f"[INFO] В справочнике ib_goods_types: {len(goods_catalog_vecs)} позиций с валидными векторами."
    )
    debug_lines.append(
        f"[INFO] В справочнике ib_equipment: {len(equip_catalog_vecs)} позиций с валидными векторами."
    )

    debug_lines.append("")
    debug_lines.append(f"[EMBED] Генерируем эмбеддинги для goods_types: {len(goods_to_embed)}")
    debug_lines.append(f"[EMBED] Генерируем эмбеддинги для equipment: {len(equip_to_embed)}")

    goods_table_rows = [
        [
            str(item.ai_id),
            (item.source_text or "").replace("\n", " "),
            str(item.match_id) if item.match_id is not None else "—",
            item.match_name or "—",
            f"{item.score:.4f}" if item.score is not None else "—",
            item.note or "",
        ]
        for item in goods_matches
    ]

    if goods_matches:
        debug_lines.append("")
        debug_lines.append(separator)
        debug_lines.append("ИТОГОВОЕ СООТВЕТСТВИЕ: ТИПЫ ПРОДУКЦИИ (ai_site_goods_types → ib_goods_types)")
        debug_lines.extend(
            _render_ascii_table(
                [
                    "ai_goods_id",
                    "ai_goods_type",
                    "match_ib_id",
                    "match_ib_name",
                    "score",
                    "note",
                ],
                goods_table_rows,
                aligns=["right", "left", "right", "left", "right", "left"],
            )
        )

    equipment_table_rows = [
        [
            str(item.ai_id),
            (item.source_text or "").replace("\n", " "),
            str(item.match_id) if item.match_id is not None else "—",
            item.match_name or "—",
            f"{item.score:.4f}" if item.score is not None else "—",
            item.note or "",
        ]
        for item in equipment_matches
    ]

    if equipment_matches:
        debug_lines.append("")
        debug_lines.append(separator)
        debug_lines.append("ИТОГОВОЕ СООТВЕТСТВИЕ: ОБОРУДОВАНИЕ (ai_site_equipment → ib_equipment)")
        debug_lines.extend(
            _render_ascii_table(
                [
                    "ai_equip_id",
                    "ai_equipment",
                    "match_ib_id",
                    "match_ib_name",
                    "score",
                    "note",
                ],
                equipment_table_rows,
                aligns=["right", "left", "right", "left", "right", "left"],
            )
        )

    debug_lines.append("")
    debug_lines.append(separator)
    debug_lines.append("СВОДКА:")
    debug_lines.append(f"- CLIENT_ID: {client_id}")
    debug_lines.append(
        f"- Обработано goods_types: {summary.goods_total}, обновлено: {summary.goods_updated}"
    )
    debug_lines.append(
        f"- Обработано equipment:   {summary.equipment_total}, обновлено: {summary.equipment_updated}"
    )
    debug_lines.append(
        f"- В ib_goods_types: {summary.catalog_goods_total} позиций с векторами"
    )
    debug_lines.append(
        f"- В ib_equipment:   {summary.catalog_equipment_total} позиций с векторами"
    )
    debug_lines.append(separator)

    if goods_matches:
        debug_lines.append("")
        debug_lines.append("[ПОДБОР GOODS] Наиболее релевантные соответствия:")
        for item in goods_matches:
            if item.match_id is not None and item.match_name and item.score is not None:
                debug_lines.append(
                    f"  ai_site_goods_types(id={item.ai_id}): '{item.source_text}' → "
                    f"ib_goods_types '{item.match_name}' (score={item.score:.4f})"
                )
            else:
                reason = item.note or "нет соответствия"
                debug_lines.append(
                    f"  ai_site_goods_types(id={item.ai_id}): '{item.source_text}' → — ({reason})"
                )

    if equipment_matches:
        debug_lines.append("")
        debug_lines.append("[ПОДБОР EQUIPMENT] Наиболее релевантные соответствия:")
        for item in equipment_matches:
            if item.match_id is not None and item.match_name and item.score is not None:
                debug_lines.append(
                    f"  ai_site_equipment(id={item.ai_id}): '{item.source_text}' → "
                    f"ib_equipment '{item.match_name}' (score={item.score:.4f})"
                )
            else:
                reason = item.note or "нет соответствия"
                debug_lines.append(
                    f"  ai_site_equipment(id={item.ai_id}): '{item.source_text}' → — ({reason})"
                )

    debug_report = "\n".join(debug_lines)

    duration_ms = _tick(started)
    log.info(
        "[ib-match] done client_id=%s goods=%s/%s equipment=%s/%s duration_ms=%s",
        client_id,
        summary.goods_updated,
        summary.goods_total,
        summary.equipment_updated,
        summary.equipment_total,
        duration_ms,
    )

    return IbMatchResponse(
        client_id=client_id,
        goods=goods_matches,
        equipment=equipment_matches,
        summary=summary,
        debug_report=debug_report,
        duration_ms=duration_ms,
    )


# ======================
# Основная реализация
# ======================
async def _analyze_impl(pars_id: int, body: AnalyzeRequest) -> AnalyzeResponse:
    t0 = dt.datetime.now()
    log.info("[analyze] start pars_id=%s", pars_id)

    # ---- Конфигурация моделей ----
    mode: SyncMode = _resolve_mode(getattr(body, "sync_mode", None))
    chat_model_raw = getattr(body, "chat_model", None) or settings.CHAT_MODEL
    embed_model_raw = getattr(body, "embed_model", None) or settings.embed_model
    chat_model: str = (chat_model_raw or "").strip()
    embed_model: str = (embed_model_raw or "").strip()

    log.info("[analyze] config pars_id=%s mode=%s chat_model=%s embed_model=%s",
             pars_id, mode.value, chat_model or "∅", embed_model or "∅")

    company_id_opt: Optional[int] = _as_int(getattr(body, "company_id", None))
    if getattr(body, "company_id", None) is not None and company_id_opt is None:
        log.warning("[analyze] invalid company_id provided pars_id=%s value=%r", pars_id, body.company_id)
    if company_id_opt is not None:
        log.info("[analyze] company_id resolved pars_id=%s company_id=%s", pars_id, company_id_opt)

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
    t_prompt = dt.datetime.now()
    prompt: str = build_prompt(text_par)
    ms_prompt = _tick(t_prompt)
    log.debug("[analyze] prompt built pars_id=%s len=%s sample=%r took_ms=%s",
              pars_id, len(prompt), _clip(prompt), ms_prompt)

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

    # Эмбеддинг описания сайта (для pars_site.text_vector)
    description_text: str = str(parsed.get("DESCRIPTION") or "").strip()
    description_vector: Optional[List[float]] = None
    description_vec_literal: Optional[str] = None
    description_embed_error: Optional[str] = None
    description_embed_ms: Optional[int] = None

    if description_text:
        t_desc = dt.datetime.now()
        try:
            description_vector = await embed_single_text(description_text, embed_model)
            description_embed_ms = _tick(t_desc)
            if description_vector:
                description_vec_literal = format_pgvector(description_vector)
                log.info(
                    "[analyze] description embedding pars_id=%s dim=%s took_ms=%s",
                    pars_id,
                    len(description_vector),
                    description_embed_ms,
                )
            else:
                log.warning(
                    "[analyze] description embedding empty pars_id=%s took_ms=%s",
                    pars_id,
                    description_embed_ms,
                )
        except Exception as ex:
            description_embed_ms = _tick(t_desc)
            description_embed_error = str(ex)
            log.error(
                "[analyze] description embedding failed pars_id=%s took_ms=%s error=%s",
                pars_id,
                description_embed_ms,
                ex,
                exc_info=True,
            )
    else:
        log.info("[analyze] description empty pars_id=%s — skip embedding", pars_id)

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
    goods_type_source: str = str(parsed.get("GOODS_TYPE_SOURCE") or "GOODS_TYPE")
    equip_source_list: List[str] = cast(List[str], parsed.get("EQUIPMENT_LIST", []) or [])
    log.info(
        "[analyze] sources pars_id=%s goods_src=%s equip_src=%s goods_origin=%s",
        pars_id,
        len(goods_source_list),
        len(equip_source_list),
        goods_type_source,
    )
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
        "source_details": {
            "goods_type_origin": goods_type_source,
            "goods_raw_count": len(cast(List[str], parsed.get("GOODS_LIST", []) or [])),
        },
        "description_embedding": {
            "text_len": len(description_text),
            "vector_dim": len(description_vector) if description_vector else 0,
            "computed": bool(description_vec_literal),
            "took_ms": description_embed_ms,
            "error": description_embed_error,
        },
        "company_id": company_id_opt,
    }
    log.info("[analyze] write phase pars_id=%s dry_run=%s mode=%s", pars_id, is_dry, mode.value)

    async def write_all_action(conn) -> Dict[str, Any]:
        t_w = dt.datetime.now()
        log.info("[analyze/write] begin pars_id=%s", pars_id)

        # 1) Сначала гарантируем наличие колонки description
        added = await repo.ensure_description_column(conn)
        log.info("[analyze/write] ensure_description_column pars_id=%s added=%s", pars_id, added)

        # 1а) Гарантируем колонку text_vector
        text_vec_added = await repo.ensure_pars_text_vector_column(conn)
        log.info(
            "[analyze/write] ensure_pars_text_vector_column pars_id=%s added=%s",
            pars_id,
            text_vec_added,
        )

        # 2) Гарантируем родителя в pars_site (или SKIP на secondary, если company_id NOT NULL)
        pars_site_result = await repo.ensure_pars_site_row(
            conn=conn,
            pars_id=pars_id,
            text_par=text_par,
            description=description_text,
            company_id=company_id_opt,
        )
        log.info(
            "[analyze/write] ensure_pars_site_row pars_id=%s status=%s",
            pars_id,
            pars_site_result.status,
        )
        if pars_site_result.status == "skipped":
            log.warning(
                "[analyze/write] pars_site insert skipped pars_id=%s reason=%s",
                pars_id,
                pars_site_result.reason,
            )

        updated = False
        text_vector_action = "skip"
        text_vector_updated = False
        prodclass_row_id: Optional[int] = None
        prodclass_result = None
        eq_rows: List[Tuple[int, str]] = []
        gt_rows: List[Tuple[int, str]] = []

        if pars_site_result.status != "skipped":
            # 3) Обновим description (если строка есть — обновится; если нет — будет False)
            updated = await repo.update_pars_description(conn, pars_id, description_text)
            log.info("[analyze/write] update_pars_description pars_id=%s updated=%s", pars_id, updated)

            # 3а) Обновим text_vector, если удалось получить эмбеддинг, либо очистим при пустом описании
            if description_vec_literal is not None:
                text_vector_updated = await repo.update_pars_text_vector(conn, pars_id, description_vec_literal)
                text_vector_action = "set"
            elif not description_text:
                text_vector_updated = await repo.update_pars_text_vector(conn, pars_id, None)
                text_vector_action = "clear"
            else:
                log.info(
                    "[analyze/write] update_pars_text_vector pars_id=%s skipped (no vector)",
                    pars_id,
                )
            log.info(
                "[analyze/write] update_pars_text_vector pars_id=%s action=%s updated=%s",
                pars_id,
                text_vector_action,
                text_vector_updated,
            )

            # 4) Жёсткая валидация prodclass
            if prodclass_id_opt is None:
                log.error("[analyze/write] prodclass_id missing pars_id=%s", pars_id)
                raise ValueError("PRODCLASS отсутствует или не является целым числом")
            if prodclass_score_opt is None:
                log.error("[analyze/write] prodclass_score missing pars_id=%s", pars_id)
                raise ValueError("PRODCLASS_SCORE отсутствует или не является числом")

            # 5) prodclass
            prodclass_result = await repo.insert_ai_site_prodclass(
                conn, pars_id, prodclass_id_opt, float(prodclass_score_opt)
            )
            prodclass_row_id = prodclass_result.row_id
            if prodclass_result.status == "skipped":
                log.warning(
                    "[analyze/write] insert_ai_site_prodclass skipped pars_id=%s id=%s reason=%s",
                    pars_id,
                    prodclass_id_opt,
                    prodclass_result.reason,
                )
            else:
                log.info(
                    "[analyze/write] insert_ai_site_prodclass pars_id=%s row_id=%s id=%s score=%s ensure_status=%s",
                    pars_id,
                    prodclass_row_id,
                    prodclass_id_opt,
                    prodclass_score_opt,
                    prodclass_result.ensure_status,
                )

            # 6) enriched вставки
            eq_rows = await repo.insert_ai_site_equipment_enriched(conn, pars_id, equip_enriched)
            log.info(
                "[analyze/write] insert_ai_site_equipment_enriched pars_id=%s inserted=%s",
                pars_id,
                len(eq_rows),
            )

            gt_rows = await repo.insert_ai_site_goods_types_enriched(conn, pars_id, goods_enriched)
            log.info(
                "[analyze/write] insert_ai_site_goods_types_enriched pars_id=%s inserted=%s",
                pars_id,
                len(gt_rows),
            )

        equip_rows_fmt = [{"id": rid, "equipment": name} for rid, name in (eq_rows or [])]
        goods_rows_fmt = [{"id": rid, "goods_type": name} for rid, name in (gt_rows or [])]

        ms_write = _tick(t_w)
        log.info("[analyze/write] done pars_id=%s took_ms=%s", pars_id, ms_write)

        return {
            "added_description_column": added,
            "added_pars_text_vector_column": text_vec_added,
            "updated_pars_site_description": updated,
            "updated_pars_site_text_vector": text_vector_updated,
            "pars_site_text_vector_action": text_vector_action,
            "ai_site_prodclass_row_id": prodclass_row_id,
            "prodclass_status": getattr(prodclass_result, "status", "skipped"),
            "prodclass_ensure_status": getattr(prodclass_result, "ensure_status", "skipped"),
            "prodclass_skip_reason": getattr(prodclass_result, "reason", None),
            "prodclass_score": prodclass_score_opt,
            "prodclass_score_source": parsed.get("PRODCLASS_SCORE_SOURCE"),
            "equipment_rows": equip_rows_fmt,
            "goods_type_rows": goods_rows_fmt,
            "pars_site_status": pars_site_result.status,
            "pars_site_skip_reason": pars_site_result.reason,
            "mirror_skipped": (pars_site_result.status == "skipped"),
            "company_id_used": company_id_opt,
        }

    try:
        if not is_dry:
            primary: Optional[AsyncEngine] = primary_opt
            secondary: Optional[AsyncEngine] = secondary_opt

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
                        second_report_obj: Any = await run_on_engine(second_engine, write_all_action)
                        ms_db2 = _tick(t_db2)
                        second_report: Dict[str, Any] = cast(Dict[str, Any], second_report_obj)
                        report["secondary_report"] = second_report
                        if second_report.get("mirror_skipped"):
                            report["secondary_mirrored"] = False
                            skip_reason = second_report.get("pars_site_skip_reason")
                            if skip_reason:
                                report["secondary_error"] = skip_reason
                            else:
                                report["secondary_error"] = "pars_site mirror skipped"
                            log.warning(
                                "[analyze/write] DUAL_WRITE mirror skipped pars_id=%s took_ms=%s reason=%s",
                                pars_id,
                                ms_db2,
                                skip_reason,
                            )
                        else:
                            report["secondary_mirrored"] = True
                            log.info(
                                "[analyze/write] DUAL_WRITE mirror ok pars_id=%s took_ms=%s",
                                pars_id,
                                ms_db2,
                            )
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
        # кейс со смешанными плейсхолдерами (:vec vs %(name)s)
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
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Analyze failed for pars_id=%s: %s", pars_id, e)
        raise HTTPException(status_code=400, detail=str(e))

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


# ======================
# Роуты
# ======================

@router.post("/v1/analyze/json", response_model=AnalyzeFromJsonResponse)
async def analyze_from_json(body: AnalyzeFromJsonRequest) -> AnalyzeFromJsonResponse:
    """Анализ без доступа к БД: на вход получаем текст и каталоги, на выходе — готовый JSON."""

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
        raise HTTPException(status_code=502, detail="Парсер ответа LLM вернул некорректную структуру")

    parsed: Dict[str, Any] = cast(Dict[str, Any], parsed_obj)
    log.info(
        "[analyze/json] parse complete keys=%s took_ms=%s",
        sorted(parsed.keys()),
        timings["parse_ms"],
    )
    log.debug(
        "[analyze/json] parsed snippet %r",
        {k: parsed.get(k) for k in ("DESCRIPTION", "PRODCLASS", "PRODCLASS_SCORE", "EQUIPMENT_LIST")},
    )

    prodclass_id = _as_int(parsed.get("PRODCLASS"))
    prodclass_score = _as_float(parsed.get("PRODCLASS_SCORE"))
    score_source = str(parsed.get("PRODCLASS_SCORE_SOURCE") or "model_reply")

    if prodclass_id is None:
        log.error("[analyze/json] prodclass missing")
        raise HTTPException(status_code=400, detail="PRODCLASS отсутствует или не является целым числом")
    if prodclass_score is None:
        log.error("[analyze/json] prodclass_score missing")
        raise HTTPException(status_code=400, detail="PRODCLASS_SCORE отсутствует или не является числом")

    prodclass_title = IB_PRODCLASS.get(prodclass_id, "—")
    prodclass_payload = ProdclassPayload(
        id=prodclass_id,
        score=float(prodclass_score),
        title=prodclass_title,
        score_source=score_source,
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
        except Exception as ex:
            description_ms = _tick(t_desc)
            log.error(
                "[analyze/json] description embedding failed took_ms=%s error=%s",
                description_ms,
                ex,
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
        goods_source_list, goods_catalog, embed_model, MATCH_THRESHOLD_GOODS
    )
    timings["goods_enrich_ms"] = _tick(t_enrich_goods)

    t_enrich_equip = dt.datetime.now()
    equip_enriched = await enrich_by_catalog(
        equip_source_list, equip_catalog, embed_model, MATCH_THRESHOLD_EQUIPMENT
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

    counts = {
        "goods_source": len(goods_source_list),
        "equip_source": len(equip_source_list),
        "goods_enriched": len(goods_payload),
        "equip_enriched": len(equip_payload),
    }

    parsed_with_title = parsed | {"PRODCLASS_TITLE": prodclass_title}

    db_payload = DbPayload(
        description=description_text,
        description_vector=description_payload,
        prodclass=prodclass_payload,
        goods_types=goods_payload,
        equipment=equip_payload,
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
        prompt=(prompt if body.return_prompt else None),
        prompt_len=len(prompt),
        answer_raw=(answer if body.return_answer_raw else None),
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


@router.post("/v1/analyze/{pars_id}", response_model=AnalyzeResponse)
async def analyze(pars_id: int, body: AnalyzeRequest):
    """Запуск анализа по известному pars_id."""
    return await _analyze_impl(pars_id, body)


@router.post("/v1/analyze/by-site/{site}", response_model=AnalyzeResponse)
async def analyze_by_site(site: str, body: AnalyzeRequest):
    """
    Анализ по домену/URL, где `site` передаётся как path-параметр.
    Примеры: /v1/analyze/by-site/elteza.ru
             /v1/analyze/by-site/https%3A%2F%2Felteza.ru%2Fcontacts
    """
    site_norm = _normalize_site(site)
    log.info("[analyze/by-site] raw=%r normalized=%r", site, site_norm)
    if not site_norm:
        raise HTTPException(status_code=400, detail="site пуст или не распознан")

    # Выбираем движок для чтения (как в analyze)
    primary_opt: Optional[AsyncEngine] = get_primary_engine()
    secondary_opt: Optional[AsyncEngine] = get_secondary_engine()
    read_engine_opt: Optional[AsyncEngine] = primary_opt or secondary_opt
    if read_engine_opt is None:
        raise HTTPException(status_code=500, detail="Нет доступных подключений к БД")
    read_engine: AsyncEngine = cast(AsyncEngine, read_engine_opt)

    # Ищем pars_id по domain_1 или по хосту url
    try:
        pars_id_obj = await run_on_engine(
            read_engine,
            lambda conn: repo.find_pars_id_by_site(conn, site_norm)
        )
    except Exception as e:
        log.exception("[analyze/by-site] search error for %r: %s", site_norm, e)
        raise HTTPException(status_code=400, detail=f"Ошибка поиска по site: {e}")

    pars_id: Optional[int] = _as_int(pars_id_obj)
    if not pars_id:
        raise HTTPException(status_code=404, detail=f"pars_site с доменом '{site_norm}' не найден")

    # Запускаем основной пайплайн ровно как для /{pars_id}
    return await _analyze_impl(pars_id, body)


@router.post("/v1/site-pipeline", response_model=SitePipelineResponse)
async def run_site_pipeline(body: SitePipelineRequest) -> SitePipelineResponse:
    """Запускает полный цикл: parse-site → analyze → ib-match → equipment-selection."""

    started = dt.datetime.now()
    requested_inn = (body.inn or "").strip() or None
    requested_site = (body.site or "").strip() or None
    site_norm = _normalize_site(requested_site) if requested_site else None

    log.info(
        "[site-pipeline] start inn=%s site=%s site_norm=%s pars_id=%s client_id=%s",
        requested_inn,
        requested_site,
        site_norm,
        body.pars_site_id,
        body.client_id,
    )

    primary = get_primary_engine()
    secondary = get_secondary_engine()
    read_engine = primary or secondary
    if read_engine is None:
        raise HTTPException(status_code=500, detail="Нет доступных подключений к БД")

    async def _resolve(conn):
        client_id: Optional[int] = body.client_id
        pars_id: Optional[int] = body.pars_site_id
        resolved_inn: Optional[str] = requested_inn
        resolved_site_norm: Optional[str] = site_norm
        client_row: Optional[dict] = None
        pars_row: Optional[dict] = None

        if pars_id is not None:
            pars_row = await repo.fetch_pars_site_basic(conn, pars_id)
            if not pars_row:
                raise HTTPException(status_code=404, detail=f"pars_site с id={pars_id} не найден")
            company_id = pars_row.get("company_id")
            if client_id is None and company_id is not None:
                try:
                    client_id = int(company_id)
                except (TypeError, ValueError):
                    client_id = None

        if client_id is not None:
            client_row = await repo.fetch_client_request(conn, client_id)
            if not client_row:
                raise HTTPException(status_code=404, detail=f"Клиент с id={client_id} не найден")
            resolved_inn = resolved_inn or client_row.get("inn")
            domain_1 = client_row.get("domain_1")
            if not resolved_site_norm and domain_1:
                resolved_site_norm = _normalize_site(str(domain_1))
            if pars_id is None:
                latest = await repo.find_latest_pars_site_for_client(conn, client_id)
                if latest:
                    pars_row = pars_row or latest
                    pars_id = latest.get("id") if latest.get("id") is not None else pars_id

        if resolved_inn and client_row is None:
            candidate = await repo.find_client_by_inn(conn, resolved_inn)
            if candidate:
                client_row = candidate
                cid = candidate.get("id")
                if cid is not None:
                    try:
                        client_id = int(cid)
                    except (TypeError, ValueError):
                        client_id = None
                if not resolved_site_norm and candidate.get("domain_1"):
                    resolved_site_norm = _normalize_site(str(candidate.get("domain_1")))

        if resolved_site_norm:
            if pars_id is None:
                found_pars = await repo.find_pars_id_by_site(conn, resolved_site_norm)
                if found_pars:
                    pars_id = found_pars
            if client_row is None:
                candidate = await repo.find_client_by_domain(conn, resolved_site_norm)
                if candidate:
                    client_row = candidate
                    cid = candidate.get("id")
                    if cid is not None:
                        try:
                            client_id = int(cid)
                        except (TypeError, ValueError):
                            client_id = None
                    if not resolved_inn and candidate.get("inn"):
                        resolved_inn = candidate.get("inn")

        if pars_id is not None and pars_row is None:
            pars_row = await repo.fetch_pars_site_basic(conn, pars_id)
            if not pars_row:
                raise HTTPException(status_code=404, detail=f"pars_site с id={pars_id} не найден")
            company_id = pars_row.get("company_id")
            if client_id is None and company_id is not None:
                try:
                    client_id = int(company_id)
                except (TypeError, ValueError):
                    client_id = None

        if client_id is not None and client_row is None:
            client_row = await repo.fetch_client_request(conn, client_id)
            if not client_row:
                raise HTTPException(status_code=404, detail=f"Клиент с id={client_id} не найден")
            if not resolved_inn and client_row.get("inn"):
                resolved_inn = client_row.get("inn")
            if not resolved_site_norm and client_row.get("domain_1"):
                resolved_site_norm = _normalize_site(str(client_row.get("domain_1")))

        if pars_id is None:
            raise HTTPException(status_code=404, detail="pars_site не найден по переданным параметрам")
        if client_id is None:
            raise HTTPException(status_code=404, detail="client_id не найден по переданным параметрам")

        pars_candidates = await repo.list_pars_sites_for_client(conn, int(client_id))

        return {
            "client_id": int(client_id),
            "pars_id": int(pars_id),
            "inn": resolved_inn,
            "site_norm": resolved_site_norm,
            "client": client_row,
            "pars": pars_row,
            "pars_candidates": pars_candidates,
        }

    try:
        resolved = await run_on_engine(read_engine, _resolve)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("[site-pipeline] resolve error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ошибка получения исходных данных: {exc}")

    client_model = ClientRow(**resolved["client"]) if resolved.get("client") else None
    pars_model = ParsSiteInfo(**resolved["pars"]) if resolved.get("pars") else None
    pars_candidates = [ParsSiteInfo(**row) for row in resolved.get("pars_candidates", [])]

    parse_result = SitePipelineParseResult(
        client=client_model,
        pars_site=pars_model,
        pars_site_candidates=pars_candidates,
    )

    resolved_section = SitePipelineResolved(
        requested_inn=requested_inn,
        requested_site=requested_site,
        normalized_site=resolved.get("site_norm"),
        inn=resolved.get("inn"),
        pars_site_id=resolved["pars_id"],
        client_id=resolved["client_id"],
    )

    analyze_result: AnalyzeResponse | None = None
    if body.run_analyze:
        analyze_payload = (body.analyze or AnalyzeRequest()).copy(update={"company_id": resolved_section.client_id})
        analyze_result = await _analyze_impl(resolved_section.pars_site_id, analyze_payload)

    ib_match_result: IbMatchResponse | None = None
    if body.run_ib_match:
        ib_options = body.ib_match or SitePipelineIbMatchOptions()
        ib_request = IbMatchRequest(
            client_id=resolved_section.client_id,
            reembed_if_exists=ib_options.reembed_if_exists,
            sync_mode=ib_options.sync_mode,
        )
        ib_match_result = await assign_ib_matches(ib_request)

    equipment_result: EquipmentSelectionResponse | None = None
    if body.run_equipment_selection:
        equipment_result = await equipment_selection(client_request_id=resolved_section.client_id)

    duration_ms = _tick(started)
    log.info(
        "[site-pipeline] done client_id=%s pars_id=%s duration_ms=%s analyze=%s ib=%s equip=%s",
        resolved_section.client_id,
        resolved_section.pars_site_id,
        duration_ms,
        bool(analyze_result),
        bool(ib_match_result),
        bool(equipment_result),
    )

    return SitePipelineResponse(
        resolved=resolved_section,
        parse_site=parse_result,
        analyze=analyze_result,
        ib_match=ib_match_result,
        equipment_selection=equipment_result,
    )


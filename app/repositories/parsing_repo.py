from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Iterable, Literal

from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.config import settings
from app.models.ib_prodclass import IB_PRODCLASS

log = logging.getLogger("repo.parsing")


@dataclass(frozen=True)
class EnsureParsSiteRowResult:
    """Результат ensure_pars_site_row."""

    status: Literal["updated", "inserted", "skipped"]
    reason: Optional[str] = None

# ---------- лог-утилиты ----------

def _clip(s: Optional[str], n: int = 200) -> str:
    """Обрезает длинные строки для логов, показывая только начало."""
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n] + f"... ({len(s)} chars)"

def _tick(t0: dt.datetime) -> int:
    """Мс с момента t0."""
    return int((dt.datetime.now() - t0).total_seconds() * 1000)


# ---------- infra / base ----------


@dataclass(frozen=True)
class _ColumnMeta:
    nullable: bool
    has_default: bool


@dataclass(frozen=True)
class EnsureIbProdclassRowResult:
    status: Literal["exists", "inserted", "skipped"]
    reason: Optional[str] = None


@dataclass(frozen=True)
class ProdclassWriteResult:
    status: Literal["inserted", "skipped"]
    ensure_status: Literal["exists", "inserted", "skipped"]
    row_id: Optional[int] = None
    reason: Optional[str] = None


_ib_prodclass_columns_cache: Optional[Dict[str, _ColumnMeta]] = None


async def _get_ib_prodclass_columns(conn: AsyncConnection) -> Dict[str, _ColumnMeta]:
    global _ib_prodclass_columns_cache
    if _ib_prodclass_columns_cache is not None:
        return _ib_prodclass_columns_cache

    q = text(
        """
        SELECT column_name, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'ib_prodclass';
        """
    )
    res = await conn.execute(q)
    columns: Dict[str, _ColumnMeta] = {}
    for row in res.mappings():
        name = str(row["column_name"])
        columns[name] = _ColumnMeta(
            nullable=str(row["is_nullable"]).upper() == "YES",
            has_default=row["column_default"] is not None,
        )

    _ib_prodclass_columns_cache = columns
    return columns


async def ensure_description_column(conn: AsyncConnection) -> bool:
    t0 = dt.datetime.now()
    log.info("[repo] ensure_description_column: check existence")
    q = text("""
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name='pars_site' AND column_name='description';
    """)
    res = await conn.execute(q)
    exists = res.first() is not None
    if not exists:
        log.info("[repo] ensure_description_column: adding column public.pars_site.description")
        await conn.execute(text("ALTER TABLE public.pars_site ADD COLUMN description TEXT;"))
        ms = _tick(t0)
        log.info("[repo] ensure_description_column: added took_ms=%s", ms)
        return True
    ms = _tick(t0)
    log.info("[repo] ensure_description_column: already exists took_ms=%s", ms)
    return False


async def ensure_pars_text_vector_column(conn: AsyncConnection) -> bool:
    """Гарантирует наличие колонки text_vector в public.pars_site."""
    t0 = dt.datetime.now()
    log.info("[repo] ensure_pars_text_vector_column: check existence")
    q = text(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name='pars_site' AND column_name='text_vector';
        """
    )
    res = await conn.execute(q)
    exists = res.first() is not None
    if exists:
        ms = _tick(t0)
        log.info("[repo] ensure_pars_text_vector_column: already exists took_ms=%s", ms)
        return False

    dim = int(getattr(settings, "VECTOR_DIM", 0) or 0)
    column_type = f"vector({dim})" if dim > 0 else "vector"
    log.info(
        "[repo] ensure_pars_text_vector_column: adding column public.pars_site.text_vector type=%s",
        column_type,
    )
    await conn.execute(
        text(f"ALTER TABLE public.pars_site ADD COLUMN text_vector {column_type};")
    )
    ms = _tick(t0)
    log.info("[repo] ensure_pars_text_vector_column: added took_ms=%s", ms)
    return True


async def fetch_text_par(conn: AsyncConnection, pars_id: int) -> str:
    t0 = dt.datetime.now()
    log.info("[repo] fetch_text_par: id=%s", pars_id)
    res = await conn.execute(text("SELECT text_par FROM public.pars_site WHERE id=:id;"), {"id": pars_id})
    row = res.first()
    if not row or not row[0]:
        ms = _tick(t0)
        log.error("[repo] fetch_text_par: not found id=%s took_ms=%s", pars_id, ms)
        raise ValueError(f"text_par не найден для pars_site.id={pars_id}")
    text_par = str(row[0])
    ms = _tick(t0)
    log.debug("[repo] fetch_text_par: id=%s len=%s sample=%r took_ms=%s", pars_id, len(text_par), _clip(text_par), ms)
    return text_par


async def update_pars_description(conn: AsyncConnection, pars_id: int, description: str) -> bool:
    t0 = dt.datetime.now()
    log.info("[repo] update_pars_description: id=%s len=%s sample=%r", pars_id, len(description or ""), _clip(description))
    res = await conn.execute(
        text("UPDATE public.pars_site SET description=:d WHERE id=:id;"),
        {"d": description, "id": pars_id},
    )
    ok = res.rowcount == 1
    ms = _tick(t0)
    log.info("[repo] update_pars_description: id=%s updated=%s took_ms=%s", pars_id, ok, ms)
    return ok


async def update_pars_text_vector(conn: AsyncConnection, pars_id: int, vec_literal: Optional[str]) -> bool:
    """
    Обновляет pars_site.text_vector. Если vec_literal=None — обнуляет значение.
    vec_literal должен быть строкой в формате pgvector: "[0.1,0.2,...]".
    """
    t0 = dt.datetime.now()
    action = "clear" if vec_literal is None else "set"
    log.info(
        "[repo] update_pars_text_vector: id=%s action=%s", pars_id, action
    )
    if vec_literal is None:
        res = await conn.execute(
            text("UPDATE public.pars_site SET text_vector = NULL WHERE id = :id;"),
            {"id": pars_id},
        )
    else:
        res = await conn.execute(
            text(
                "UPDATE public.pars_site SET text_vector = CAST(:vec AS vector) WHERE id = :id;"
            ),
            {"id": pars_id, "vec": vec_literal},
        )
    ok = res.rowcount == 1
    ms = _tick(t0)
    log.info("[repo] update_pars_text_vector: id=%s updated=%s took_ms=%s", pars_id, ok, ms)
    return ok


async def ensure_ib_prodclass_row(
    conn: AsyncConnection, prodclass_id: int
) -> EnsureIbProdclassRowResult:
    """Гарантирует наличие записи в справочнике ib_prodclass."""

    t0 = dt.datetime.now()
    log.info("[repo] ensure_ib_prodclass_row: id=%s", prodclass_id)

    res = await conn.execute(
        text("SELECT 1 FROM public.ib_prodclass WHERE id = :id;"),
        {"id": prodclass_id},
    )
    if res.first() is not None:
        ms = _tick(t0)
        log.info(
            "[repo] ensure_ib_prodclass_row: id=%s already exists took_ms=%s",
            prodclass_id,
            ms,
        )
        return EnsureIbProdclassRowResult(status="exists")

    title = IB_PRODCLASS.get(prodclass_id)
    if title is None:
        ms = _tick(t0)
        log.error(
            "[repo] ensure_ib_prodclass_row: id=%s missing in IB_PRODCLASS took_ms=%s",
            prodclass_id,
            ms,
        )
        raise ValueError(
            f"Prodclass {prodclass_id} отсутствует в локальном справочнике IB_PRODCLASS"
        )

    columns = await _get_ib_prodclass_columns(conn)
    required_extra_cols = [
        name
        for name, meta in columns.items()
        if name not in {"id", "prodclass"} and not meta.nullable and not meta.has_default
    ]
    if required_extra_cols:
        ms = _tick(t0)
        missing = ", ".join(sorted(required_extra_cols))
        log.warning(
            "[repo] ensure_ib_prodclass_row: id=%s cannot autoinsert required_columns=%s took_ms=%s",
            prodclass_id,
            missing,
            ms,
        )
        reason = (
            "Автоматическое добавление prodclass недоступно: таблица public.ib_prodclass "
            f"требует значения в колонках {missing}. Добавьте запись {prodclass_id} вручную "
            "или настройте источники данных для этих полей."
        )
        log.warning(
            "[repo] ensure_ib_prodclass_row: id=%s skip auto-insert reason=%s", prodclass_id, reason
        )
        return EnsureIbProdclassRowResult(status="skipped", reason=reason)

    try:
        await conn.execute(
            text(
                """
                INSERT INTO public.ib_prodclass (id, prodclass)
                VALUES (:id, :title)
                ON CONFLICT (id) DO NOTHING;
                """
            ),
            {"id": prodclass_id, "title": title},
        )
    except IntegrityError as exc:
        ms = _tick(t0)
        log.error(
            "[repo] ensure_ib_prodclass_row: id=%s insert failed took_ms=%s error=%s",
            prodclass_id,
            ms,
            exc,
            exc_info=True,
        )
        raise ValueError(
            f"Не удалось добавить prodclass {prodclass_id} в ib_prodclass: {exc}"
        ) from exc

    ms = _tick(t0)
    log.info(
        "[repo] ensure_ib_prodclass_row: id=%s inserted took_ms=%s",
        prodclass_id,
        ms,
    )
    return EnsureIbProdclassRowResult(status="inserted")


async def ensure_pars_site_row(
    conn: AsyncConnection,
    pars_id: int,
    text_par: str,
    description: str,
    *,
    company_id: Optional[int] = None,
) -> EnsureParsSiteRowResult:
    """
    Идемпотентное обеспечение наличия строки в public.pars_site для данного id.
    Стратегия «безопасного зеркалирования»:
      1) UPDATE
      2) Если строки нет — проверяем company_id: если NOT NULL без DEFAULT,
         то требуется явный company_id (например, из тела запроса).
         Иначе INSERT (id, text_par, description[, company_id]) + ON CONFLICT UPDATE description.
    Возвращает EnsureParsSiteRowResult со статусом:
        * "updated"  — строка уже существовала;
        * "inserted" — строка была добавлена;
        * "skipped"  — вставка невозможна (например, нет company_id для NOT NULL).
    """
    t0 = dt.datetime.now()
    log.info("[repo] ensure_pars_site_row: id=%s", pars_id)

    upd = await conn.execute(
        text("UPDATE public.pars_site SET description=:d WHERE id=:id;"),
        {"d": description, "id": pars_id},
    )
    if upd.rowcount and upd.rowcount > 0:
        took = _tick(t0)
        log.info("[repo] ensure_pars_site_row: id=%s existed=True took_ms=%s", pars_id, took)
        return EnsureParsSiteRowResult(status="updated")

    # Проверяем ограничение на company_id
    meta = await conn.execute(
        text("""
            SELECT is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name='pars_site' AND column_name='company_id'
            LIMIT 1;
        """)
    )
    row = meta.first()

    company_nullable = True
    company_has_default = False
    if row:
        is_nullable, col_default = (row[0], row[1])
        company_nullable = (str(is_nullable).upper() == "YES")
        company_has_default = (col_default is not None)

    requires_company = (not company_nullable and not company_has_default)

    if requires_company and company_id is None:
        took = _tick(t0)
        reason = (
            "pars_site.company_id is NOT NULL w/o DEFAULT; mirror requires company mapping"
        )
        log.warning(
            "[repo] ensure_pars_site_row: id=%s insert SKIPPED — %s took_ms=%s",
            pars_id,
            reason,
            took,
        )
        return EnsureParsSiteRowResult(status="skipped", reason=reason)

    columns = ["id", "text_par", "description"]
    values = [":id", ":tp", ":d"]
    params = {"id": pars_id, "tp": text_par, "d": description}

    if company_id is not None:
        columns.append("company_id")
        values.append(":cid")
        params["cid"] = company_id
        log.info(
            "[repo] ensure_pars_site_row: id=%s using company_id=%s for insert",
            pars_id,
            company_id,
        )

    query = text(
        """
            INSERT INTO public.pars_site ({columns})
            VALUES ({values})
            ON CONFLICT (id) DO UPDATE
            SET description = EXCLUDED.description
        """.format(columns=", ".join(columns), values=", ".join(values))
    )

    try:
        await conn.execute(query, params)
    except IntegrityError as exc:
        sqlstate = getattr(getattr(exc, "orig", None), "sqlstate", None)
        took = _tick(t0)
        if sqlstate == "23503":
            cid_suffix = f"={company_id}" if company_id is not None else ""
            reason = f"foreign key violation for pars_site.company_id{cid_suffix}"
            log.warning(
                "[repo] ensure_pars_site_row: id=%s insert skipped due to FK constraint took_ms=%s",
                pars_id,
                took,
            )
            log.debug("[repo] ensure_pars_site_row: id=%s fk error=%s", pars_id, exc)
            return EnsureParsSiteRowResult(status="skipped", reason=reason)
        log.error(
            "[repo] ensure_pars_site_row: id=%s insert failed integrity error took_ms=%s error=%s",
            pars_id,
            took,
            exc,
        )
        raise

    took = _tick(t0)
    log.info("[repo] ensure_pars_site_row: id=%s inserted=True took_ms=%s", pars_id, took)
    return EnsureParsSiteRowResult(status="inserted")


# ---------- helpers for catalogs ----------

async def _table_columns(conn: AsyncConnection, table: str) -> list[tuple[str, str, Optional[str]]]:
    t0 = dt.datetime.now()
    log.info("[repo] _table_columns: table=%s", table)
    q = text("""
        SELECT column_name, data_type, udt_name
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name=:t
        ORDER BY ordinal_position
    """)
    res = await conn.execute(q, {"t": table})
    rows = res.fetchall()
    ms = _tick(t0)
    log.debug("[repo] _table_columns: table=%s columns=%s took_ms=%s",
              table, [r[0] for r in rows], ms)
    return [(str(r[0]), str(r[1]), (str(r[2]) if r[2] is not None else None)) for r in rows]


async def _detect_text_column(conn: AsyncConnection, table: str, candidates: list[str]) -> str:
    t0 = dt.datetime.now()
    log.info("[repo] _detect_text_column: table=%s candidates=%s", table, candidates)
    cols = await _table_columns(conn, table)
    names = {c[0].lower(): c[0] for c in cols}
    for cand in candidates:
        if cand.lower() in names:
            col = names[cand.lower()]
            ms = _tick(t0)
            log.info("[repo] _detect_text_column: table=%s matched=%s took_ms=%s", table, col, ms)
            return col
    for col, dtype, _ in cols:
        if dtype in ("text", "character varying", "varchar"):
            ms = _tick(t0)
            log.info("[repo] _detect_text_column: table=%s auto=%s dtype=%s took_ms=%s", table, col, dtype, ms)
            return col
    ms = _tick(t0)
    log.error("[repo] _detect_text_column: table=%s NOT FOUND took_ms=%s", table, ms)
    raise RuntimeError(f"Не найдено текстовой колонки в public.{table}")


async def _detect_vector_column(conn: AsyncConnection, table: str, candidates: list[str]) -> Optional[str]:
    t0 = dt.datetime.now()
    log.info("[repo] _detect_vector_column: table=%s candidates=%s", table, candidates)
    cols = await _table_columns(conn, table)
    names = {c[0].lower(): c[0] for c in cols}
    for cand in candidates:
        if cand.lower() in names:
            col = names[cand.lower()]
            ms = _tick(t0)
            log.info("[repo] _detect_vector_column: table=%s matched=%s took_ms=%s", table, col, ms)
            return col
    for col, dtype, udt in cols:
        if dtype == "USER-DEFINED" and (udt or "").lower() == "vector":
            ms = _tick(t0)
            log.info("[repo] _detect_vector_column: table=%s auto=%s took_ms=%s", table, col, ms)
            return col
    ms = _tick(t0)
    log.info("[repo] _detect_vector_column: table=%s vector NOT FOUND took_ms=%s", table, ms)
    return None


def _as_pgvector_literal(vec: Any) -> Optional[str]:
    log.debug("[repo] _as_pgvector_literal: input_type=%s", type(vec).__name__)
    if vec is None:
        return None

    if isinstance(vec, (list, tuple)):
        try:
            arr = [float(x) for x in vec]
            return "[" + ", ".join(f"{x:.12g}" for x in arr) + "]"
        except Exception:
            pass

    if isinstance(vec, str):
        s = vec.strip()
        if s.startswith("[") and s.endswith("]"):
            return s
        if "," in s:
            return "[" + s + "]"
        try:
            float(s)
            return "[" + s + "]"
        except Exception:
            return None

    if isinstance(vec, Iterable) and not isinstance(vec, (str, bytes)):
        try:
            arr = [float(x) for x in vec]
            return "[" + ", ".join(f"{x:.12g}" for x in arr) + "]"
        except Exception:
            return None

    return None


# ---------- catalogs ----------

async def fetch_goods_types_catalog(conn: AsyncConnection) -> list[dict]:
    t0 = dt.datetime.now()
    table = settings.IB_GOODS_TYPES_TABLE or "ib_goods_types"
    id_col = settings.IB_GOODS_TYPES_ID_COLUMN or "id"
    log.info("[repo] fetch_goods_types_catalog: table=%s id_col=%s", table, id_col)

    name_col = settings.IB_GOODS_TYPES_NAME_COLUMN or await _detect_text_column(
        conn, table, ["goods_type_name", "name", "goods_type", "title", "label"]
    )
    vec_col = settings.IB_GOODS_TYPES_VECTOR_COLUMN
    if vec_col is None:
        vec_col = await _detect_vector_column(conn, table, ["goods_type_vector", "vector", "emb", "embedding"])

    log.info("[repo] fetch_goods_types_catalog: table=%s name_col=%s vec_col=%s", table, name_col, vec_col)

    company_filter = ""
    params: dict[str, Any] = {}
    if settings.IB_GOODS_TYPES_COMPANY_ID is not None:
        company_filter = 'WHERE "company_id" = :cid'
        params["cid"] = settings.IB_GOODS_TYPES_COMPANY_ID
        log.info("[repo] fetch_goods_types_catalog: filter company_id=%s", params["cid"])

    if vec_col:
        q = text(
            f'SELECT "{id_col}" AS id, "{name_col}" AS name, "{vec_col}"::text AS vec '
            f'FROM public.{table} {company_filter} '
            f'ORDER BY "{id_col}";'
        )
    else:
        q = text(
            f'SELECT "{id_col}" AS id, "{name_col}" AS name, NULL::text AS vec '
            f'FROM public.{table} {company_filter} '
            f'ORDER BY "{id_col}";'
        )
    res = await conn.execute(q, params)
    rows = res.fetchall()
    out: list[dict] = []
    for r in rows:
        out.append({"id": int(r[0]), "name": str(r[1]), "vec": (str(r[2]) if r[2] is not None else None)})
    ms = _tick(t0)
    log.info("[repo] fetch_goods_types_catalog: table=%s rows=%s took_ms=%s", table, len(out), ms)
    log.debug("[repo] fetch_goods_types_catalog: sample=%r", out[:3])
    return out


async def fetch_equipment_catalog(conn: AsyncConnection) -> list[dict]:
    t0 = dt.datetime.now()
    table = settings.IB_EQUIPMENT_TABLE or "ib_equipment"
    id_col = settings.IB_EQUIPMENT_ID_COLUMN or "id"
    log.info("[repo] fetch_equipment_catalog: table=%s id_col=%s", table, id_col)

    name_col = settings.IB_EQUIPMENT_NAME_COLUMN or await _detect_text_column(
        conn, table, ["equipment_name", "name", "equipment", "title", "label"]
    )
    vec_col = settings.IB_EQUIPMENT_VECTOR_COLUMN
    if vec_col is None:
        vec_col = await _detect_vector_column(conn, table, ["equipment_vector", "vector", "emb", "embedding"])

    log.info("[repo] fetch_equipment_catalog: table=%s name_col=%s vec_col=%s", table, name_col, vec_col)

    company_filter = ""
    params: dict[str, Any] = {}
    if settings.IB_EQUIPMENT_COMPANY_ID is not None:
        company_filter = 'WHERE "company_id" = :cid'
        params["cid"] = settings.IB_EQUIPMENT_COMPANY_ID
        log.info("[repo] fetch_equipment_catalog: filter company_id=%s", params["cid"])

    if vec_col:
        q = text(
            f'SELECT "{id_col}" AS id, "{name_col}" AS name, "{vec_col}"::text AS vec '
            f'FROM public.{table} {company_filter} '
            f'ORDER BY "{id_col}";'
        )
    else:
        q = text(
            f'SELECT "{id_col}" AS id, "{name_col}" AS name, NULL::text AS vec '
            f'FROM public.{table} {company_filter} '
            f'ORDER BY "{id_col}";'
        )
    res = await conn.execute(q, params)
    rows = res.fetchall()
    out: list[dict] = []
    for r in rows:
        out.append({"id": int(r[0]), "name": str(r[1]), "vec": (str(r[2]) if r[2] is not None else None)})
    ms = _tick(t0)
    log.info("[repo] fetch_equipment_catalog: table=%s rows=%s took_ms=%s", table, len(out), ms)
    log.debug("[repo] fetch_equipment_catalog: sample=%r", out[:3])
    return out


async def fetch_client_goods(conn: AsyncConnection, client_id: int) -> list[dict]:
    t0 = dt.datetime.now()
    log.info("[repo] fetch_client_goods: client_id=%s", client_id)
    q = text(
        """
        SELECT g.id, g.goods_type, g.text_vector::text
        FROM public.ai_site_goods_types g
        JOIN public.pars_site ps ON g.text_par_id = ps.id
        WHERE ps.company_id = :cid
        ORDER BY g.id;
        """
    )
    res = await conn.execute(q, {"cid": client_id})
    rows = res.fetchall()
    out: list[dict] = []
    for row in rows:
        out.append(
            {
                "id": int(row[0]),
                "text": str(row[1]) if row[1] is not None else "",
                "vec": str(row[2]) if row[2] is not None else None,
            }
        )
    ms = _tick(t0)
    log.info("[repo] fetch_client_goods: client_id=%s rows=%s took_ms=%s", client_id, len(out), ms)
    log.debug("[repo] fetch_client_goods: sample=%r", out[:3])
    return out


async def fetch_client_equipment(conn: AsyncConnection, client_id: int) -> list[dict]:
    t0 = dt.datetime.now()
    log.info("[repo] fetch_client_equipment: client_id=%s", client_id)
    q = text(
        """
        SELECT e.id, e.equipment, e.text_vector::text
        FROM public.ai_site_equipment e
        JOIN public.pars_site ps ON e.text_pars_id = ps.id
        WHERE ps.company_id = :cid
        ORDER BY e.id;
        """
    )
    res = await conn.execute(q, {"cid": client_id})
    rows = res.fetchall()
    out: list[dict] = []
    for row in rows:
        out.append(
            {
                "id": int(row[0]),
                "text": str(row[1]) if row[1] is not None else "",
                "vec": str(row[2]) if row[2] is not None else None,
            }
        )
    ms = _tick(t0)
    log.info("[repo] fetch_client_equipment: client_id=%s rows=%s took_ms=%s", client_id, len(out), ms)
    log.debug("[repo] fetch_client_equipment: sample=%r", out[:3])
    return out


async def update_goods_vectors(conn: AsyncConnection, items: list[tuple[int, str]]) -> int:
    if not items:
        return 0
    t0 = dt.datetime.now()
    log.info("[repo] update_goods_vectors: items=%s", len(items))
    total = 0
    q = text(
        """
        UPDATE public.ai_site_goods_types
        SET text_vector = :vec::vector
        WHERE id = :id;
        """
    )
    for gid, vec in items:
        res = await conn.execute(q, {"id": gid, "vec": vec})
        if res.rowcount:
            total += int(res.rowcount)
    ms = _tick(t0)
    log.info("[repo] update_goods_vectors: updated=%s took_ms=%s", total, ms)
    return total


async def update_equipment_vectors(conn: AsyncConnection, items: list[tuple[int, str]]) -> int:
    if not items:
        return 0
    t0 = dt.datetime.now()
    log.info("[repo] update_equipment_vectors: items=%s", len(items))
    total = 0
    q = text(
        """
        UPDATE public.ai_site_equipment
        SET text_vector = :vec::vector
        WHERE id = :id;
        """
    )
    for eid, vec in items:
        res = await conn.execute(q, {"id": eid, "vec": vec})
        if res.rowcount:
            total += int(res.rowcount)
    ms = _tick(t0)
    log.info("[repo] update_equipment_vectors: updated=%s took_ms=%s", total, ms)
    return total


async def update_goods_matches(conn: AsyncConnection, items: list[tuple[int, int, float]]) -> int:
    if not items:
        return 0
    t0 = dt.datetime.now()
    log.info("[repo] update_goods_matches: items=%s", len(items))
    total = 0
    q = text(
        """
        UPDATE public.ai_site_goods_types
        SET goods_type_ID = :match_id, goods_types_score = :score
        WHERE id = :id;
        """
    )
    for gid, match_id, score in items:
        res = await conn.execute(q, {"id": gid, "match_id": match_id, "score": score})
        if res.rowcount:
            total += int(res.rowcount)
    ms = _tick(t0)
    log.info("[repo] update_goods_matches: updated=%s took_ms=%s", total, ms)
    return total


async def update_equipment_matches(conn: AsyncConnection, items: list[tuple[int, int, float]]) -> int:
    if not items:
        return 0
    t0 = dt.datetime.now()
    log.info("[repo] update_equipment_matches: items=%s", len(items))
    total = 0
    q = text(
        """
        UPDATE public.ai_site_equipment
        SET equipment_ID = :match_id, equipment_score = :score
        WHERE id = :id;
        """
    )
    for eid, match_id, score in items:
        res = await conn.execute(q, {"id": eid, "match_id": match_id, "score": score})
        if res.rowcount:
            total += int(res.rowcount)
    ms = _tick(t0)
    log.info("[repo] update_equipment_matches: updated=%s took_ms=%s", total, ms)
    return total


# ---------- inserts (enriched) ----------

async def insert_ai_site_prodclass(
    conn: AsyncConnection, pars_id: int, prodclass: int, score: float
) -> ProdclassWriteResult:
    t0 = dt.datetime.now()
    log.info("[repo] insert_ai_site_prodclass: pars_id=%s prodclass=%s score=%s", pars_id, prodclass, score)
    ensure_result = await ensure_ib_prodclass_row(conn, prodclass)
    if ensure_result.status == "skipped":
        ms_skip = _tick(t0)
        log.warning(
            "[repo] insert_ai_site_prodclass: skip pars_id=%s prodclass=%s reason=%s took_ms=%s",
            pars_id,
            prodclass,
            ensure_result.reason,
            ms_skip,
        )
        return ProdclassWriteResult(
            status="skipped",
            ensure_status=ensure_result.status,
            reason=ensure_result.reason,
        )
    try:
        res = await conn.execute(
            text(
                """
                INSERT INTO public.ai_site_prodclass (text_pars_id, prodclass, prodclass_score)
                VALUES (:pid, :pc, :sc)
                RETURNING id;
                """
            ),
            {"pid": pars_id, "pc": prodclass, "sc": score},
        )
    except IntegrityError as exc:
        ms = _tick(t0)
        log.error(
            "[repo] insert_ai_site_prodclass: integrity error pars_id=%s prodclass=%s took_ms=%s error=%s",
            pars_id,
            prodclass,
            ms,
            exc,
            exc_info=True,
        )
        raise ValueError(
            "Не удалось сохранить prodclass: проверьте наличие записи в ib_prodclass"
        ) from exc
    new_id = int(res.scalar_one())
    ms = _tick(t0)
    log.info("[repo] insert_ai_site_prodclass: inserted id=%s took_ms=%s", new_id, ms)
    return ProdclassWriteResult(
        status="inserted",
        ensure_status=ensure_result.status,
        row_id=new_id,
    )


async def insert_ai_site_equipment_enriched(
    conn: AsyncConnection,
    pars_id: int,
    items: list[dict],  # {"text","match_id","score","vec"|"vec_str"}
) -> list[tuple[int, str]]:
    t0 = dt.datetime.now()
    log.info("[repo] insert_ai_site_equipment_enriched: pars_id=%s items=%s", pars_id, len(items))
    out: list[tuple[int, str]] = []
    for idx, it in enumerate(items, 1):
        raw_vec = it.get("vec", it.get("vec_str"))
        vec_lit = _as_pgvector_literal(raw_vec)  # "[...]" или None

        if vec_lit is None:
            res = await conn.execute(
                text("""
                    INSERT INTO public.ai_site_equipment
                        (text_pars_id, equipment, equipment_score, equipment_ID, text_vector)
                    VALUES
                        (:pid, :eq, :sc, :eid, NULL)
                    RETURNING id, equipment;
                """),
                {"pid": pars_id, "eq": it["text"], "sc": it["score"], "eid": it["match_id"]},
            )
        else:
            res = await conn.execute(
                text("""
                    INSERT INTO public.ai_site_equipment
                        (text_pars_id, equipment, equipment_score, equipment_ID, text_vector)
                    VALUES
                        (:pid, :eq, :sc, :eid, CAST(:vec AS vector))
                    RETURNING id, equipment;
                """),
                {"pid": pars_id, "eq": it["text"], "sc": it["score"], "eid": it["match_id"], "vec": vec_lit},
            )

        row = res.first()
        if row:
            out.append((int(row[0]), str(row[1])))

    ms = _tick(t0)
    log.info("[repo] insert_ai_site_equipment_enriched: inserted=%s took_ms=%s", len(out), ms)
    log.debug("[repo] insert_ai_site_equipment_enriched: sample=%r", out[:3])
    return out


async def insert_ai_site_goods_types_enriched(
    conn: AsyncConnection,
    pars_id: int,
    items: list[dict],  # {"text","match_id","score","vec"|"vec_str"}
) -> list[tuple[int, str]]:
    t0 = dt.datetime.now()
    log.info("[repo] insert_ai_site_goods_types_enriched: pars_id=%s items=%s", pars_id, len(items))
    out: list[tuple[int, str]] = []
    for idx, it in enumerate(items, 1):
        raw_vec = it.get("vec", it.get("vec_str"))
        vec_lit = _as_pgvector_literal(raw_vec)

        if vec_lit is None:
            res = await conn.execute(
                text("""
                    INSERT INTO public.ai_site_goods_types
                        (text_par_id, goods_type, goods_types_score, goods_type_ID, text_vector)
                    VALUES
                        (:pid, :gt, :sc, :gid, NULL)
                    RETURNING id, goods_type;
                """),
                {"pid": pars_id, "gt": it["text"], "sc": it["score"], "gid": it["match_id"]},
            )
        else:
            res = await conn.execute(
                text("""
                    INSERT INTO public.ai_site_goods_types
                        (text_par_id, goods_type, goods_types_score, goods_type_ID, text_vector)
                    VALUES
                        (:pid, :gt, :sc, :gid, CAST(:vec AS vector))
                    RETURNING id, goods_type;
                """),
                {"pid": pars_id, "gt": it["text"], "sc": it["score"], "gid": it["match_id"], "vec": vec_lit},
            )

        row = res.first()
        if row:
            out.append((int(row[0]), str(row[1])))

    ms = _tick(t0)
    log.info("[repo] insert_ai_site_goods_types_enriched: inserted=%s took_ms=%s", len(out), ms)
    log.debug("[repo] insert_ai_site_goods_types_enriched: sample=%r", out[:3])
    return out


# ---------- by-site lookup ----------

async def find_pars_id_by_site(conn: AsyncConnection, site_or_host: str) -> Optional[int]:
    """
    Ищем pars_site.id:
      1) точное совпадение по domain_1 (без www)
      2) по хосту, извлечённому из url (host без www)
    """
    host = (site_or_host or "").lower().strip()

    # 1) Совпадение по domain_1
    q1 = text("""
        SELECT id
        FROM public.pars_site
        WHERE lower(regexp_replace(domain_1, '^www\\.', '')) = :h
        ORDER BY id ASC
        LIMIT 1
    """)
    r1 = await conn.execute(q1, {"h": host})
    row = r1.first()
    if row and row[0]:
        return int(row[0])

    # 2) Совпадение по url → host (обрезаем схему/путь/www)
    q2 = text("""
        SELECT id
        FROM public.pars_site
        WHERE lower(regexp_replace(
            regexp_replace(
                regexp_replace(url, '^https?://', ''), -- без протокола
            '/.*$', ''),                               -- без пути
        '^www\\.', '')) = :h
        ORDER BY id ASC
        LIMIT 1
    """)
    r2 = await conn.execute(q2, {"h": host})
    row2 = r2.first()
    return int(row2[0]) if row2 and row2[0] else None


async def fetch_pars_site_basic(conn: AsyncConnection, pars_id: int) -> Optional[dict]:
    """Возвращает базовую информацию о pars_site по идентификатору."""

    t0 = dt.datetime.now()
    log.info("[repo] fetch_pars_site_basic: pars_id=%s", pars_id)
    query = text(
        """
        SELECT id, company_id, domain_1, url
        FROM public.pars_site
        WHERE id = :pid
        LIMIT 1;
        """
    )
    res = await conn.execute(query, {"pid": pars_id})
    row = res.mappings().first()
    ms = _tick(t0)
    if row:
        data = dict(row)
        log.info("[repo] fetch_pars_site_basic: pars_id=%s found company_id=%s took_ms=%s", pars_id, data.get("company_id"), ms)
        return data
    log.warning("[repo] fetch_pars_site_basic: pars_id=%s not_found took_ms=%s", pars_id, ms)
    return None


async def find_latest_pars_site_for_client(conn: AsyncConnection, client_id: int) -> Optional[dict]:
    """Возвращает последнюю запись pars_site для указанного клиента (по id DESC)."""

    t0 = dt.datetime.now()
    log.info("[repo] find_latest_pars_site_for_client: client_id=%s", client_id)
    query = text(
        """
        SELECT id, company_id, domain_1, url
        FROM public.pars_site
        WHERE company_id = :cid
        ORDER BY id DESC
        LIMIT 1;
        """
    )
    res = await conn.execute(query, {"cid": client_id})
    row = res.mappings().first()
    ms = _tick(t0)
    if row:
        data = dict(row)
        log.info("[repo] find_latest_pars_site_for_client: client_id=%s pars_id=%s took_ms=%s", client_id, data.get("id"), ms)
        return data
    log.warning("[repo] find_latest_pars_site_for_client: client_id=%s not_found took_ms=%s", client_id, ms)
    return None


async def list_pars_sites_for_client(conn: AsyncConnection, client_id: int, *, limit: int = 50) -> list[dict]:
    """Возвращает несколько pars_site для клиента (по убыванию id)."""

    t0 = dt.datetime.now()
    log.info("[repo] list_pars_sites_for_client: client_id=%s limit=%s", client_id, limit)
    query = text(
        """
        SELECT id, company_id, domain_1, url
        FROM public.pars_site
        WHERE company_id = :cid
        ORDER BY id DESC
        LIMIT :limit;
        """
    )
    res = await conn.execute(query, {"cid": client_id, "limit": max(1, limit)})
    rows = [dict(row) for row in res.mappings().all()]
    ms = _tick(t0)
    log.info(
        "[repo] list_pars_sites_for_client: client_id=%s rows=%s took_ms=%s",
        client_id,
        len(rows),
        ms,
    )
    return rows


async def fetch_client_request(conn: AsyncConnection, client_id: int) -> Optional[dict]:
    """Возвращает информацию о записи clients_requests."""

    t0 = dt.datetime.now()
    log.info("[repo] fetch_client_request: client_id=%s", client_id)
    query = text(
        """
        SELECT id, company_name, inn, domain_1, started_at, ended_at
        FROM public.clients_requests
        WHERE id = :cid
        LIMIT 1;
        """
    )
    res = await conn.execute(query, {"cid": client_id})
    row = res.mappings().first()
    ms = _tick(t0)
    if row:
        data = dict(row)
        log.info("[repo] fetch_client_request: client_id=%s found inn=%s took_ms=%s", client_id, data.get("inn"), ms)
        return data
    log.warning("[repo] fetch_client_request: client_id=%s not_found took_ms=%s", client_id, ms)
    return None


async def find_client_by_inn(conn: AsyncConnection, inn: str) -> Optional[dict]:
    """Ищет клиента в clients_requests по ИНН."""

    inn_norm = (inn or "").strip()
    if not inn_norm:
        return None

    t0 = dt.datetime.now()
    log.info("[repo] find_client_by_inn: inn=%s", inn_norm)
    query = text(
        """
        SELECT id, company_name, inn, domain_1, started_at, ended_at
        FROM public.clients_requests
        WHERE inn = :inn
        ORDER BY id DESC
        LIMIT 1;
        """
    )
    res = await conn.execute(query, {"inn": inn_norm})
    row = res.mappings().first()
    ms = _tick(t0)
    if row:
        data = dict(row)
        log.info(
            "[repo] find_client_by_inn: inn=%s client_id=%s took_ms=%s",
            inn_norm,
            data.get("id"),
            ms,
        )
        return data
    log.warning("[repo] find_client_by_inn: inn=%s not_found took_ms=%s", inn_norm, ms)
    return None


async def find_client_by_domain(conn: AsyncConnection, domain: str) -> Optional[dict]:
    """Ищет клиента в clients_requests по домену (без www)."""

    host = (domain or "").strip().lower()
    if not host:
        return None

    t0 = dt.datetime.now()
    log.info("[repo] find_client_by_domain: domain=%s", host)
    query = text(
        """
        SELECT id, company_name, inn, domain_1, started_at, ended_at
        FROM public.clients_requests
        WHERE lower(regexp_replace(domain_1, '^www\\.', '')) = :domain
        ORDER BY id DESC
        LIMIT 1;
        """
    )
    res = await conn.execute(query, {"domain": host})
    row = res.mappings().first()
    ms = _tick(t0)
    if row:
        data = dict(row)
        log.info(
            "[repo] find_client_by_domain: domain=%s client_id=%s took_ms=%s",
            host,
            data.get("id"),
            ms,
        )
        return data
    log.warning("[repo] find_client_by_domain: domain=%s not_found took_ms=%s", host, ms)
    return None

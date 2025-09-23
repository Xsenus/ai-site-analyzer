from __future__ import annotations
from typing import Any, Optional
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy import text
from app.config import settings

# ---------- infra / base ----------

async def ensure_description_column(conn: AsyncConnection) -> bool:
    q = text("""
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name='pars_site' AND column_name='description';
    """)
    res = await conn.execute(q)
    exists = res.first() is not None
    if not exists:
        await conn.execute(text("ALTER TABLE public.pars_site ADD COLUMN description TEXT;"))
        return True
    return False


async def fetch_text_par(conn: AsyncConnection, pars_id: int) -> str:
    res = await conn.execute(text("SELECT text_par FROM public.pars_site WHERE id=:id;"), {"id": pars_id})
    row = res.first()
    if not row or not row[0]:
        raise ValueError(f"text_par не найден для pars_site.id={pars_id}")
    return row[0]


async def update_pars_description(conn: AsyncConnection, pars_id: int, description: str) -> bool:
    res = await conn.execute(
        text("UPDATE public.pars_site SET description=:d WHERE id=:id;"),
        {"d": description, "id": pars_id},
    )
    return res.rowcount == 1


# ---------- helpers ----------

async def _table_columns(conn: AsyncConnection, table: str) -> list[tuple[str, str, Optional[str]]]:
    """
    Возвращает [(column_name, data_type, udt_name | None)].
    Для vector в PG: data_type='USER-DEFINED', udt_name='vector'
    """
    q = text("""
        SELECT column_name, data_type, udt_name
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name=:t
        ORDER BY ordinal_position
    """)
    res = await conn.execute(q, {"t": table})
    return [(str(r[0]), str(r[1]), (str(r[2]) if r[2] is not None else None)) for r in res.fetchall()]


async def _detect_text_column(conn: AsyncConnection, table: str, candidates: list[str]) -> str:
    cols = await _table_columns(conn, table)
    names = {c[0].lower(): c[0] for c in cols}
    for cand in candidates:
        if cand.lower() in names:
            return names[cand.lower()]
    for col, dtype, _ in cols:
        if dtype in ("text", "character varying", "varchar"):
            return col
    raise RuntimeError(f"Не найдено текстовой колонки в public.{table}")


async def _detect_vector_column(conn: AsyncConnection, table: str, candidates: list[str]) -> Optional[str]:
    cols = await _table_columns(conn, table)
    names = {c[0].lower(): c[0] for c in cols}
    for cand in candidates:
        if cand.lower() in names:
            return names[cand.lower()]
    # авто: ищем data_type='USER-DEFINED' и udt_name='vector'
    for col, dtype, udt in cols:
        if dtype == "USER-DEFINED" and (udt or "").lower() == "vector":
            return col
    return None


# ---------- catalogs (возвращают список dict: {id, name, vec?}) ----------

async def fetch_goods_types_catalog(conn: AsyncConnection) -> list[dict]:
    table = settings.IB_GOODS_TYPES_TABLE or "ib_goods_types"
    id_col = settings.IB_GOODS_TYPES_ID_COLUMN or "id"
    name_col = settings.IB_GOODS_TYPES_NAME_COLUMN or await _detect_text_column(
        conn, table, ["goods_type_name", "name", "goods_type", "title", "label"]
    )
    vec_col = settings.IB_GOODS_TYPES_VECTOR_COLUMN
    if vec_col is None:
        vec_col = await _detect_vector_column(conn, table, ["goods_type_vector", "vector", "emb", "embedding"])

    company_filter = ""
    params: dict[str, Any] = {}
    if settings.IB_GOODS_TYPES_COMPANY_ID is not None:
        company_filter = 'WHERE "company_id" = :cid'
        params["cid"] = settings.IB_GOODS_TYPES_COMPANY_ID

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
    out: list[dict] = []
    for r in res.fetchall():
        out.append({"id": int(r[0]), "name": str(r[1]), "vec": (str(r[2]) if r[2] is not None else None)})
    return out


async def fetch_equipment_catalog(conn: AsyncConnection) -> list[dict]:
    table = settings.IB_EQUIPMENT_TABLE or "ib_equipment"
    id_col = settings.IB_EQUIPMENT_ID_COLUMN or "id"
    name_col = settings.IB_EQUIPMENT_NAME_COLUMN or await _detect_text_column(
        conn, table, ["equipment_name", "name", "equipment", "title", "label"]
    )
    vec_col = settings.IB_EQUIPMENT_VECTOR_COLUMN
    if vec_col is None:
        vec_col = await _detect_vector_column(conn, table, ["equipment_vector", "vector", "emb", "embedding"])

    company_filter = ""
    params: dict[str, Any] = {}
    if settings.IB_EQUIPMENT_COMPANY_ID is not None:
        company_filter = 'WHERE "company_id" = :cid'
        params["cid"] = settings.IB_EQUIPMENT_COMPANY_ID

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
    out: list[dict] = []
    for r in res.fetchall():
        out.append({"id": int(r[0]), "name": str(r[1]), "vec": (str(r[2]) if r[2] is not None else None)})
    return out


# ---------- inserts (enriched) ----------

async def insert_ai_site_prodclass(conn: AsyncConnection, pars_id: int, prodclass: int, score: float) -> int:
    res = await conn.execute(
        text("""
            INSERT INTO public.ai_site_prodclass (text_pars_id, prodclass, prodclass_score)
            VALUES (:pid, :pc, :sc)
            RETURNING id;
        """),
        {"pid": pars_id, "pc": prodclass, "sc": score},
    )
    return int(res.scalar_one())


async def insert_ai_site_equipment_enriched(
    conn: AsyncConnection,
    pars_id: int,
    items: list[dict],  # {text, match_id, score, vec_str}
) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for it in items:
        res = await conn.execute(
            text("""
                INSERT INTO public.ai_site_equipment
                    (text_pars_id, equipment, equipment_score, equipment_ID, text_vector)
                VALUES
                    (:pid, :eq, :sc, :eid, :vec::vector)
                RETURNING id, equipment;
            """),
            {"pid": pars_id, "eq": it["text"], "sc": it["score"], "eid": it["match_id"], "vec": it["vec_str"]},
        )
        row = res.first()
        if row:
            out.append((int(row[0]), str(row[1])))
    return out


async def insert_ai_site_goods_types_enriched(
    conn: AsyncConnection,
    pars_id: int,
    items: list[dict],  # {text, match_id, score, vec_str}
) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for it in items:
        res = await conn.execute(
            text("""
                INSERT INTO public.ai_site_goods_types
                    (text_par_id, goods_type, goods_types_score, goods_type_ID, text_vector)
                VALUES
                    (:pid, :gt, :sc, :gid, :vec::vector)
                RETURNING id, goods_type;
            """),
            {"pid": pars_id, "gt": it["text"], "sc": it["score"], "gid": it["match_id"], "vec": it["vec_str"]},
        )
        row = res.first()
        if row:
            out.append((int(row[0]), str(row[1])))
    return out

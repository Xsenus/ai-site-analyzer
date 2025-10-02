from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.schemas.equipment_selection import (
    ClientRow,
    EquipmentAllRow,
    EquipmentDetailRow,
    EquipmentGoodsLinkRow,
    EquipmentSelectionResponse,
    EquipmentWayRow,
    GoodsTypeRow,
    GoodsTypeScoreRow,
    ProdclassDetail,
    ProdclassSourceRow,
    SiteEquipmentRow,
    WorkshopRow,
)


@dataclass
class _WayTable:
    """Вспомогательная структура для массовой записи итоговых таблиц."""

    name: str
    rows: List[EquipmentWayRow]


def _to_float(value: Any) -> Optional[float]:
    """Осторожно конвертирует входное значение в float, если это возможно."""
    if value is None:
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_score(value: float, digits: int = 4) -> float:
    """Единая функция округления, чтобы везде получить одинаковую точность."""

    return float(f"{value:.{digits}f}")


async def _drop_and_create_table(conn: AsyncConnection, table: str) -> None:
    """Переинициализирует итоговую таблицу в БД перед записью свежих данных."""
    await conn.execute(text(f'DROP TABLE IF EXISTS "{table}";'))
    await conn.execute(
        text(
            f'''
            CREATE TABLE "{table}"(
                id              BIGINT,
                equipment_name  TEXT,
                score           NUMERIC(8,4)
            );
            '''
        )
    )


async def _insert_way_rows(
    conn: AsyncConnection, table: str, rows: Iterable[EquipmentWayRow]
) -> None:
    """Построчно записывает результаты в подготовленную таблицу."""
    for row in rows:
        await conn.execute(
            text(f'INSERT INTO "{table}"(id, equipment_name, score) VALUES (:id, :equipment_name, :score);'),
            {
                "id": row.id,
                "equipment_name": row.equipment_name,
                "score": row.score,
            },
        )


def _build_client(row: Optional[Dict[str, Any]]) -> Optional[ClientRow]:
    """Конструирует Pydantic-модель клиента из словаря БД."""
    if row is None:
        return None
    return ClientRow(**row)


def _build_goods_types(rows: List[Dict[str, Any]]) -> List[GoodsTypeRow]:
    """Подготавливает список типов продукции с приведением числовых полей."""
    result: List[GoodsTypeRow] = []
    for row in rows:
        if row.get("goods_types_score") is not None:
            row["goods_types_score"] = _to_float(row["goods_types_score"])
        result.append(GoodsTypeRow(**row))
    return result


def _build_site_equipment(rows: List[Dict[str, Any]]) -> List[SiteEquipmentRow]:
    """Подготавливает список оборудования, найденного на сайте клиента."""
    result: List[SiteEquipmentRow] = []
    for row in rows:
        if row.get("equipment_score") is not None:
            row["equipment_score"] = _to_float(row["equipment_score"])
        result.append(SiteEquipmentRow(**row))
    return result


def _build_prodclass_rows(rows: List[Dict[str, Any]]) -> List[ProdclassSourceRow]:
    """Приводит сырые данные по prodclass к типизированной структуре."""
    result: List[ProdclassSourceRow] = []
    for row in rows:
        if row.get("prodclass_score") is not None:
            row["prodclass_score"] = _to_float(row["prodclass_score"])
        result.append(ProdclassSourceRow(**row))
    return result


async def compute_equipment_selection(
    conn: AsyncConnection,
    client_request_id: int,
) -> EquipmentSelectionResponse:
    # --- 0. Загружаем основную информацию о клиенте ---
    client_row = await conn.execute(
        text(
            """
            SELECT id, company_name, inn, domain_1, started_at, ended_at
            FROM clients_requests
            WHERE id = :cid;
            """
        ),
        {"cid": client_request_id},
    )
    client_mapping = client_row.mappings().first()
    client = _build_client(dict(client_mapping)) if client_mapping else None

    # --- 1. Типы продукции, найденные на сайте клиента ---
    goods_types_res = await conn.execute(
        text(
            """
            SELECT
                gst.id,
                gst.goods_type,
                gst.goods_type_id,
                gst.goods_types_score,
                pst.id AS text_par_id,
                pst.url,
                gst.created_at
            FROM ai_site_goods_types AS gst
            JOIN pars_site AS pst ON pst.id = gst.text_par_id
            WHERE pst.company_id = :cid
            ORDER BY gst.created_at, gst.id;
            """
        ),
        {"cid": client_request_id},
    )
    goods_types_rows = [dict(row) for row in goods_types_res.mappings().all()]
    goods_types = _build_goods_types(goods_types_rows)

    # --- 2. Оборудование, найденное на сайте клиента ---
    site_eq_res = await conn.execute(
        text(
            """
            SELECT
                eq.id,
                eq.equipment,
                eq.equipment_id,
                eq.equipment_score,
                pst.id AS text_pars_id,
                pst.url,
                eq.created_at
            FROM ai_site_equipment AS eq
            JOIN pars_site AS pst ON pst.id = eq.text_pars_id
            WHERE pst.company_id = :cid
            ORDER BY eq.created_at, eq.id;
            """
        ),
        {"cid": client_request_id},
    )
    site_equipment_rows = [dict(row) for row in site_eq_res.mappings().all()]
    site_equipment = _build_site_equipment(site_equipment_rows)

    # --- 3. Prodclass-ы, полученные из парсинга сайта ---
    prodclass_res = await conn.execute(
        text(
            """
            SELECT
                ap.id            AS ai_row_id,
                ap.prodclass     AS prodclass_id,
                ip.prodclass     AS prodclass_name,
                ap.prodclass_score,
                ap.text_pars_id,
                pst.url,
                ap.created_at
            FROM ai_site_prodclass AS ap
            JOIN pars_site      AS pst ON pst.id = ap.text_pars_id
            JOIN ib_prodclass   AS ip  ON ip.id = ap.prodclass
            WHERE pst.company_id = :cid
            ORDER BY ap.created_at, ap.id;
            """
        ),
        {"cid": client_request_id},
    )
    prodclass_rows_raw = [dict(row) for row in prodclass_res.mappings().all()]
    prodclass_rows = _build_prodclass_rows(prodclass_rows_raw)

    # Готовим сгруппированные наборы оценок по каждому prodclass.
    prodclass_groups: Dict[tuple[int, Optional[str]], List[float]] = defaultdict(list)
    for row in prodclass_rows_raw:
        score = _to_float(row.get("prodclass_score"))
        if score is None:
            continue
        key = (int(row["prodclass_id"]), row.get("prodclass_name"))
        prodclass_groups[key].append(score)

    # В prodclass_details собираем расширенный лог для фронта.
    prodclass_details: List[ProdclassDetail] = []
    # e1_scores хранит лучший SCORE_E1 по каждому оборудованию.
    e1_scores: Dict[int, EquipmentDetailRow] = {}

    def _prodclass_sort_key(item: tuple[tuple[int, Optional[str]], List[float]]) -> tuple[float, int, int]:
        key, values = item
        avg = -(sum(values) / len(values)) if values else 0.0
        votes = -len(values)
        prodclass = key[0] if key else 0
        return (avg, votes, prodclass)

    # Перебираем prodclass в порядке убывания среднего SCORE, чтобы приоритетно
    # рассматривать наиболее релевантные.
    for (prodclass_id, prodclass_name), scores in sorted(
        prodclass_groups.items(), key=_prodclass_sort_key
    ):
        votes = len(scores)
        avg_score = _round_score(sum(scores) / votes) if votes else None
        # Пытаемся найти производственные цеха по прямой связи prodclass → workshops.
        workshops_res = await conn.execute(
            text(
                """
                SELECT id, workshop_name, workshop_score, prodclass_id, company_id
                FROM ib_workshops
                WHERE prodclass_id = :pcid
                ORDER BY id;
                """
            ),
            {"pcid": prodclass_id},
        )
        workshops_rows = []
        for row in workshops_res.mappings().all():
            row_dict = dict(row)
            workshops_rows.append(
                WorkshopRow(
                    id=int(row_dict.get("id")),
                    workshop_name=row_dict.get("workshop_name"),
                    workshop_score=_to_float(row_dict.get("workshop_score")),
                    prodclass_id=row_dict.get("prodclass_id"),
                    company_id=row_dict.get("company_id"),
                )
            )
        path = "direct" if workshops_rows else "fallback_none"
        fallback_industry_id: Optional[int] = None
        fallback_prodclass_ids: Optional[List[int]] = None
        fallback_workshops: Optional[List[WorkshopRow]] = None
        equipment_rows: List[EquipmentDetailRow] = []

        async def _fetch_equipment(
            workshop_ids: List[int], source: str, factor: float
        ) -> List[EquipmentDetailRow]:
            """Общая функция подсчёта SCORE_E1 для прямого и fallback-путей."""
            if not workshop_ids:
                return []
            equipment_res = await conn.execute(
                text(
                    """
                    SELECT
                        e.id,
                        e.equipment_name,
                        e.workshop_id,
                        e.equipment_score,
                        e.equipment_score_real,
                        GREATEST(e.equipment_score, COALESCE(e.equipment_score_real, 0)) AS equipment_score_max
                    FROM ib_equipment AS e
                    WHERE e.workshop_id = ANY(:ws_list)
                    ORDER BY e.id;
                    """
                ),
                {"ws_list": workshop_ids},
            )
            equip_list: List[EquipmentDetailRow] = []
            for row in equipment_res.mappings().all():
                row_dict = dict(row)
                score_max = _to_float(row_dict.get("equipment_score_max")) or 0.0
                base_score = avg_score or 0.0
                score_e1 = _round_score(base_score * factor * score_max)
                equip_list.append(
                    EquipmentDetailRow(
                        id=int(row_dict["id"]),
                        equipment_name=row_dict.get("equipment_name"),
                        workshop_id=row_dict.get("workshop_id"),
                        equipment_score=_to_float(row_dict.get("equipment_score")),
                        equipment_score_real=_to_float(row_dict.get("equipment_score_real")),
                        equipment_score_max=score_max,
                        score_e1=score_e1,
                        source=source,
                    )
                )
            return equip_list

        if workshops_rows:
            # Прямой путь: оборудование берём из найденных цехов.
            path = "direct"
            equipment_rows = await _fetch_equipment([w.id for w in workshops_rows], "direct", 1.0)
        else:
            path = "fallback_noworkshops"
            # Пытаемся расширить поиск до отрасли.
            industry_res = await conn.execute(
                text(
                    """
                    SELECT id AS prodclass_id, industry_id
                    FROM ib_prodclass
                    WHERE id = :pcid;
                    """
                ),
                {"pcid": prodclass_id},
            )
            industry_row = industry_res.mappings().first()
            if industry_row and industry_row.get("industry_id") is not None:
                industry_dict = dict(industry_row)
                fallback_industry_id = int(industry_dict["industry_id"])
                prodclass_industry_res = await conn.execute(
                    text(
                        """
                        SELECT id AS prodclass_id
                        FROM ib_prodclass
                        WHERE industry_id = :ind
                        ORDER BY id;
                        """
                    ),
                    {"ind": fallback_industry_id},
                )
                fallback_ids = [int(dict(row)["prodclass_id"]) for row in prodclass_industry_res.mappings().all()]
                fallback_prodclass_ids = fallback_ids or None
                if fallback_ids:
                    # Нашли prodclass в отрасли → ищем все workshops по ним.
                    fallback_ws_res = await conn.execute(
                        text(
                            """
                            SELECT id, workshop_name, workshop_score, prodclass_id, company_id
                            FROM ib_workshops
                            WHERE prodclass_id = ANY(:pc_list)
                            ORDER BY id;
                            """
                        ),
                        {"pc_list": fallback_ids},
                    )
                    fallback_workshops_rows = []
                    for row in fallback_ws_res.mappings().all():
                        row_dict = dict(row)
                        fallback_workshops_rows.append(
                            WorkshopRow(
                                id=int(row_dict.get("id")),
                                workshop_name=row_dict.get("workshop_name"),
                                workshop_score=_to_float(row_dict.get("workshop_score")),
                                prodclass_id=row_dict.get("prodclass_id"),
                                company_id=row_dict.get("company_id"),
                            )
                        )
                    fallback_workshops = fallback_workshops_rows or None
                    if fallback_workshops_rows:
                        # Фолбэк-расчёт: SCORE_E1 умножаем на доп. коэффициент 0.75.
                        path = "fallback"
                        equipment_rows = await _fetch_equipment(
                            [w.id for w in fallback_workshops_rows],
                            "fallback",
                            0.75,
                        )
                else:
                    path = "fallback_noworkshops"
            else:
                path = "fallback_fail"

        # Обновляем карту лучших результатов по SCORE_E1.
        for equipment in equipment_rows:
            existing = e1_scores.get(equipment.id)
            if existing is None or (equipment.score_e1 or 0) > (existing.score_e1 or 0):
                e1_scores[equipment.id] = equipment

        # Добавляем подробную информацию по текущему prodclass.
        prodclass_details.append(
            ProdclassDetail(
                prodclass_id=prodclass_id,
                prodclass_name=prodclass_name,
                score_1=avg_score,
                votes=votes,
                path=path,
                workshops=workshops_rows,
                fallback_industry_id=fallback_industry_id,
                fallback_prodclass_ids=fallback_prodclass_ids,
                fallback_workshops=fallback_workshops,
                equipment=equipment_rows,
            )
        )

    # Формируем итоговый список 1way (через prodclass) по убыванию SCORE.
    equipment_1way_rows = sorted(
        (
            EquipmentWayRow(
                id=eq_id,
                equipment_name=detail.equipment_name,
                score=detail.score_e1 or 0.0,
            )
            for eq_id, detail in e1_scores.items()
        ),
        key=lambda item: (-item.score, item.id),
    )

    goods_type_score_map: Dict[int, float] = {}
    for row in goods_types:
        if row.goods_type_id is None or row.goods_types_score is None:
            continue
        current = goods_type_score_map.get(row.goods_type_id)
        if current is None or row.goods_types_score > current:
            goods_type_score_map[row.goods_type_id] = row.goods_types_score
    # Сохраняем максимальный SCORE по каждому goods_type.
    goods_type_scores = [
        GoodsTypeScoreRow(goods_type_id=gid, crores_2=_round_score(score))
        for gid, score in sorted(goods_type_score_map.items())
    ]
    goods_type_score_lookup = {item.goods_type_id: item.crores_2 for item in goods_type_scores}

    goods_links: List[EquipmentGoodsLinkRow] = []
    equipment_2way_map: Dict[int, EquipmentWayRow] = {}
    if goods_type_scores:
        # Заготавливаем второй путь (через goods_type).
        gt_list = [item.goods_type_id for item in goods_type_scores]
        goods_links_res = await conn.execute(
            text(
                """
                SELECT
                    ieg.equipment_id AS equipment_id,
                    g.goods_type_id,
                    e.equipment_score,
                    e.equipment_name
                FROM ib_equipment_goods AS ieg
                JOIN ib_goods        AS g  ON g.id = ieg.goods_id
                JOIN ib_equipment    AS e  ON e.id = ieg.equipment_id
                WHERE g.goods_type_id = ANY(:gt_list);
                """
            ),
            {"gt_list": gt_list},
        )
        for row in goods_links_res.mappings().all():
            row_dict = dict(row)
            equipment_id = int(row_dict["equipment_id"])
            goods_type_id = int(row_dict["goods_type_id"])
            crores_2 = goods_type_score_lookup.get(goods_type_id, 0.0)
            crores_3 = _to_float(row_dict.get("equipment_score")) or 0.0
            score_e2 = _round_score(crores_2 * crores_3)
            goods_link = EquipmentGoodsLinkRow(
                equipment_id=equipment_id,
                goods_type_id=goods_type_id,
                crores_2=crores_2,
                crores_3=crores_3,
                score_e2=score_e2,
                equipment_name=row_dict.get("equipment_name"),
            )
            goods_links.append(goods_link)
            existing = equipment_2way_map.get(equipment_id)
            if existing is None or score_e2 > existing.score:
                equipment_2way_map[equipment_id] = EquipmentWayRow(
                    id=equipment_id,
                    equipment_name=row_dict.get("equipment_name"),
                    score=score_e2,
                )

    # Итоговый список для 2way.
    equipment_2way_rows = sorted(
        equipment_2way_map.values(), key=lambda item: (-item.score, item.id)
    )

    equipment_3way_map: Dict[int, EquipmentWayRow] = {}
    equipment_names: Dict[int, Optional[str]] = {}
    if site_equipment:
        # Третий путь — напрямую из ai_site_equipment.
        equipment_ids = sorted({row.equipment_id for row in site_equipment if row.equipment_id is not None})
        if equipment_ids:
            names_res = await conn.execute(
                text(
                    """
                    SELECT id, equipment_name
                    FROM ib_equipment
                    WHERE id = ANY(:ids);
                    """
                ),
                {"ids": equipment_ids},
            )
            for row in names_res.mappings().all():
                row_dict = dict(row)
                equipment_names[int(row_dict["id"])] = row_dict.get("equipment_name")
        for row in site_equipment:
            if row.equipment_id is None or row.equipment_score is None:
                continue
            eq_id = int(row.equipment_id)
            score_e3 = _round_score(row.equipment_score)
            name = row.equipment
            if eq_id in equipment_names:
                name = equipment_names[eq_id]
            existing = equipment_3way_map.get(eq_id)
            if existing is None or score_e3 > existing.score:
                equipment_3way_map[eq_id] = EquipmentWayRow(id=eq_id, equipment_name=name, score=score_e3)

    # Итоговый список для 3way.
    equipment_3way_rows = sorted(
        equipment_3way_map.values(), key=lambda item: (-item.score, item.id)
    )

    combined_map: Dict[int, EquipmentAllRow] = {}
    priority_map = {"1way": 1, "2way": 2, "3way": 3}
    for source, priority, rows in (
        ("1way", 1, equipment_1way_rows),
        ("2way", 2, equipment_2way_rows),
        ("3way", 3, equipment_3way_rows),
    ):
        # Объединяем результаты: берём максимальный SCORE, при равенстве — тот,
        # что пришёл из пути с более высоким приоритетом.
        for row in rows:
            existing = combined_map.get(row.id)
            if existing is None or row.score > existing.score or (
                math.isclose(row.score, existing.score, rel_tol=1e-9, abs_tol=1e-9)
                and priority < priority_map.get(existing.source, 99)
            ):
                combined_map[row.id] = EquipmentAllRow(
                    id=row.id,
                    equipment_name=row.equipment_name,
                    score=row.score,
                    source=source,
                )

    equipment_all_rows = sorted(
        combined_map.values(),
        key=lambda item: (-item.score, priority_map.get(item.source, 99), item.id),
    )

    for table in (
        _WayTable("EQUIPMENT_1way", equipment_1way_rows),
        _WayTable("EQUIPMENT_2way", equipment_2way_rows),
        _WayTable("EQUIPMENT_3way", equipment_3way_rows),
        _WayTable(
            "EQUIPMENT_ALL",
            [
                EquipmentWayRow(
                    id=row.id,
                    equipment_name=row.equipment_name,
                    score=row.score,
                )
                for row in equipment_all_rows
            ],
        ),
    ):
        # Перед записью в БД переинициализируем таблицу и заливаем новые значения.
        await _drop_and_create_table(conn, table.name)
        await _insert_way_rows(conn, table.name, table.rows)

    return EquipmentSelectionResponse(
        client=client,
        goods_types=goods_types,
        site_equipment=site_equipment,
        prodclass_rows=prodclass_rows,
        prodclass_details=prodclass_details,
        goods_type_scores=goods_type_scores,
        goods_links=goods_links,
        equipment_1way=equipment_1way_rows,
        equipment_2way=equipment_2way_rows,
        equipment_3way=equipment_3way_rows,
        equipment_all=equipment_all_rows,
    )

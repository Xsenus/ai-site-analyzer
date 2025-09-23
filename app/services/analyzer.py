# app/services/analyzer.py
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from app.config import settings
from app.models.ib_prodclass import IB_PRODCLASS

# Клиент OpenAI (async)
oai = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# --- настройки сопоставления ---
MATCH_THRESHOLD_EQUIPMENT: float = 0.45
MATCH_THRESHOLD_GOODS: float = 0.45
EMBED_BATCH_SIZE: int = 96


# ---------- prompt & openai ----------

def ib_prodclass_table_str() -> str:
    lines = ["ID        Название класса производства"]
    for k in sorted(IB_PRODCLASS):
        lines.append(f"{k}\t{IB_PRODCLASS[k]}")
    return "\n".join(lines)


def build_prompt(text_par: str) -> str:
    rules = """
Ниже представлен текст с сайта компании.
Напиши ответ в таком виде:
[DESCRIPTION] =[текст]
[PRODCLASS]=[ID класса производства ib_prodclass]
[PRODCLASS_SCORE]=[уровень сходства с ib_prodclass]
[EQUIPMENT_SITE]=[Оборудование1; Оборудование2; Оборудование3; и так далее]
[GOODS]=[Товар/услуга1; Товар/услуга2; Товар/услуга3; и так далее]
[GOODS_TYPE]=[ИмяТНВЭД1; ИмяТНВЭД2; ИмяТНВЭД3; и так далее]
, где
DESCRIPTION =[текст]
PRODCLASS = ID класса производства ib_prodclass, к которому компания относится ближе всего
PRODCLASS_SCORE= уровень смысловой связи (сходства) текста сайта компании с выбранным PRODCLASS; число с точкой в диапазоне 0.00–1.00.
EQUIPMENT_SITE = Список детальных наименований производственного/технологического оборудования, который упоминается на сайте и с помощью которого компания производит товары и или выполняет услуги, но не которое продает или реализует;
Пропускай наименования, которые слишком обширные, к примеру: «Промышленное оборудование», «Современная технологическая линия»,  "Высокотехнологические линии", "Современное оборудование" или близкие по смыслу. Либо добаляй к названию технологический процесс.
GOODS = Список товаров/услуг компании, который упоминается на сайте, которые она сама производит или реализует;
GOODS_TYPE = Для каждого GOODS нужно присвоить свой ТНВЭД и указать наименование ТНВЭД без кода.
Требования: текст и набор наименований для каждого параметра должен быть заключен в квадратные скобки [].
Название переменной и набор данных пишется с новой строки.
Отвечай строго без пояснений.
"""
    return f"{rules}\nТаблица ib_prodclass\n{ib_prodclass_table_str()}\nТекст с сайта компании: {text_par}"


async def call_openai(prompt: str, chat_model: str) -> str:
    resp = await oai.chat.completions.create(
        model=chat_model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Отвечай строго в 6 строках по заданному шаблону. Не добавляй пояснений."},
            {"role": "user", "content": prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


# ---------- embeddings & math ----------

async def _embeddings(texts: List[str], embed_model: str) -> List[List[float]]:
    """
    Получить эмбеддинги для списка текстов батчами.
    """
    all_vecs: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        chunk = texts[i : i + EMBED_BATCH_SIZE]
        resp = await oai.embeddings.create(model=embed_model, input=chunk)
        all_vecs.extend([list(map(float, d.embedding)) for d in resp.data])
    return all_vecs


def _cosine_lists(u: List[float], v: List[float]) -> float:
    """
    Косинусная близость двух векторов без numpy.
    Если длины различаются, считаем по минимальной общей длине.
    """
    m = min(len(u), len(v))
    if m == 0:
        return 0.0
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for i in range(m):
        a = u[i]
        b = v[i]
        dot += a * b
        nu += a * a
        nv += b * b
    if nu <= 0.0 or nv <= 0.0:
        return 0.0
    return dot / (math.sqrt(nu) * math.sqrt(nv))


def _to_vec_str(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"


def _parse_pgvector_text(s: str) -> Optional[List[float]]:
    """
    Приходит из БД как строка '(0.1,0.2,...)' или '[0.1,0.2,...]' (в зависимости от представления).
    Попробуем распарсить безопасно.
    """
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    if (s[0], s[-1]) in {("(", ")"), ("[", "]")}:
        s = s[1:-1]
    try:
        parts = s.split(",")
        out: List[float] = []
        for p in parts:
            p = p.strip()
            if p:
                out.append(float(p))
        return out if out else None
    except Exception:
        return None


# ---------- parse model reply ----------

def _extract_section(tag: str, text: str, required: bool = True) -> Optional[str]:
    m = re.search(rf"\[{re.escape(tag)}\]\s*=\s*\[(.*?)\]", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        if required:
            raise ValueError(f"Не удалось распарсить секцию [{tag}]")
        return None
    return m.group(1).strip()


def _split(s: str) -> List[str]:
    return [p.strip() for p in s.split(";") if p.strip()]


def _parse_prodclass_id(s: str) -> int:
    m = re.search(r"\d+", s)
    if not m:
        raise ValueError("В секции [PRODCLASS] не найдено целое число ID")
    return int(m.group(0))


def _parse_score(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s_norm = s.replace(",", ".")
    m = re.search(r"[-+]?\d*\.?\d+", s_norm)
    if not m:
        return None
    val = float(m.group(0))
    if 1.0 < val <= 100.0:
        val = val / 100.0
    val = max(0.0, min(1.0, val))
    return float(f"{val:.2f}")


async def parse_openai_answer(answer: str, text_par_for_fallback: str, embed_model: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    data["DESCRIPTION"] = _extract_section("DESCRIPTION", answer, required=True) or ""
    data["PRODCLASS_RAW"] = _extract_section("PRODCLASS", answer, required=True) or ""
    data["PRODCLASS_SCORE_RAW"] = _extract_section("PRODCLASS_SCORE", answer, required=False)
    data["EQUIPMENT_RAW"] = _extract_section("EQUIPMENT_SITE", answer, required=True) or ""
    data["GOODS_RAW"] = _extract_section("GOODS", answer, required=True) or ""
    data["GOODS_TYPE_RAW"] = _extract_section("GOODS_TYPE", answer, required=True) or ""

    data["PRODCLASS"] = _parse_prodclass_id(data["PRODCLASS_RAW"])
    score = _parse_score(data["PRODCLASS_SCORE_RAW"])

    used_fallback = False
    if score is None:
        cls_name = IB_PRODCLASS.get(data["PRODCLASS"], f"Класс {data['PRODCLASS']}")
        # ограничим длину текста, чтобы не уходить в слишком большие запросы
        vecs = await _embeddings([text_par_for_fallback[:6000], cls_name], embed_model)
        v1, v2 = vecs[0], vecs[1]
        sim = _cosine_lists(v1, v2)
        score = float(f"{(sim + 1.0) / 2.0:.2f}")
        used_fallback = True

    data["PRODCLASS_SCORE"] = score
    data["PRODCLASS_SCORE_SOURCE"] = "fallback_embeddings" if used_fallback else "model_reply"
    data["EQUIPMENT_LIST"] = _split(data["EQUIPMENT_RAW"])
    data["GOODS_LIST"] = _split(data["GOODS_RAW"])
    data["GOODS_TYPE_LIST"] = _split(data["GOODS_TYPE_RAW"])
    return data


# ---------- matching to catalogs (с поддержкой вектора из БД) ----------

async def enrich_by_catalog(
    items: List[str],
    catalog: List[Dict[str, Any]],  # каждый: {id, name, vec?} где vec — str|(None)
    embed_model: str,
    min_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Возвращает [{text, match_id, score, vec, vec_str}]
    - эмбеддинг для item считаем всегда
    - для каталога: используем vec из БД, если есть, иначе считаем
    """
    if not items:
        return []

    # векторы для items
    item_vecs = await _embeddings(items, embed_model)

    # векторы каталога: пробуем взять из БД (vec), если нет — считаем
    cat_ids: List[int] = [int(c["id"]) for c in catalog]
    cat_names: List[str] = [str(c["name"]) for c in catalog]

    cat_vecs: List[Optional[List[float]]] = []
    need_compute_texts: List[str] = []
    need_idx: List[int] = []

    for idx, c in enumerate(catalog):
        parsed = _parse_pgvector_text(str(c.get("vec") or ""))
        if parsed is None:
            need_compute_texts.append(cat_names[idx])
            need_idx.append(idx)
            cat_vecs.append(None)  # placeholder
        else:
            cat_vecs.append(parsed)

    if need_compute_texts:
        computed = await _embeddings(need_compute_texts, embed_model)
        ptr = 0
        for idx in need_idx:
            cat_vecs[idx] = computed[ptr]
            ptr += 1

    # сопоставление
    out: List[Dict[str, Any]] = []
    for i, text in enumerate(items):
        v_item = item_vecs[i]
        best_j = -1
        best_cos = -1.0
        for j, v_cat in enumerate(cat_vecs):
            if v_cat is None:
                continue
            cos = _cosine_lists(v_item, v_cat)
            if cos > best_cos:
                best_cos = cos
                best_j = j

        if best_j < 0:
            out.append(
                {"text": text, "match_id": None, "score": None, "vec": v_item, "vec_str": _to_vec_str(v_item)}
            )
            continue

        score = float(f"{(best_cos + 1.0) / 2.0:.2f}")
        match_id: Optional[int] = int(cat_ids[best_j]) if score >= min_threshold else None

        out.append(
            {
                "text": text,
                "match_id": match_id,
                "score": (score if match_id is not None else None),
                "vec": v_item,
                "vec_str": _to_vec_str(v_item),
            }
        )

    return out

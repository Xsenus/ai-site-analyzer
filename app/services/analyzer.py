# app/services/analyzer.py
from __future__ import annotations

import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from app.config import settings
from app.models.ib_prodclass import IB_PRODCLASS

_PRODCLASS_NAME_VECS_CACHE: Dict[str, Tuple[List[int], List[List[float]]]] = {}

# Кэш на часто встречающиеся элементы каталогов, чтобы не пересчитывать эмбеддинги.
_CATALOG_VECTOR_CACHE: "OrderedDict[Tuple[str, str], List[float]]" = OrderedDict()
_CATALOG_VECTOR_CACHE_LIMIT = 256

_EMPTY_ITEM_MARKERS = {
    "",
    "нет",
    "нет данных",
    "не указано",
    "не указаны",
    "не указана",
    "не предоставлено",
    "не предоставлены",
    "не предоставлен",
    "отсутствует",
    "отсутствуют",
    "отсутствует оборудование",
    "нет оборудования",
    "нет товар",
    "нет товара",
    "нет товаров",
    "none",
    "n/a",
    "na",
    "-",
    "—",
}

_oai_client: AsyncOpenAI | None = None


@dataclass
class _EmbeddingProdclassGuess:
    """Предсказание класса производства на основе эмбеддингов."""

    text_vector: Optional[List[float]]
    prodclass_id: Optional[int]
    score: Optional[float]


@dataclass
class _ProdclassResolution:
    """Итоговое решение по классу производства после всех проверок."""

    prodclass_id: int
    source: str
    text_vector: Optional[List[float]]
    final_score: Optional[float]
    embed_best_id: Optional[int]
    embed_best_score: Optional[float]


def _get_openai_client() -> AsyncOpenAI:
    """Return a cached AsyncOpenAI client or raise if key is missing."""

    global _oai_client

    if _oai_client is not None:
        return _oai_client

    api_key = settings.OPENAI_API_KEY
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured")

    _oai_client = AsyncOpenAI(api_key=api_key)
    return _oai_client

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
    client = _get_openai_client()
    resp = await client.chat.completions.create(
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
    client = _get_openai_client()
    all_vecs: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        chunk = texts[i : i + EMBED_BATCH_SIZE]
        resp = await client.embeddings.create(model=embed_model, input=chunk)
        all_vecs.extend([list(map(float, d.embedding)) for d in resp.data])
    return all_vecs


async def embed_single_text(text: str, embed_model: str) -> Optional[List[float]]:
    """Возвращает эмбеддинг для одного текста или None, если текст пустой."""
    if not text:
        return None
    payload = text.strip()
    if not payload:
        return None
    vecs = await _embeddings([payload], embed_model)
    if not vecs:
        return None
    return vecs[0]


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

_SECTION_BLOCK_RE = re.compile(
    r"""
    (?<!\S)                           # секция начинается с пробела или начала строки
    \[\s*(?P<tag>[A-Z0-9_]+)\s*\]   # имя секции в квадратных скобках
    (?P<content>.*?)                  # содержимое до следующей секции
    (?=(?<!\S)\[[A-Z0-9_]+\s*\]|\Z)  # остановиться перед следующей секцией или концом текста
    """,
    flags=re.IGNORECASE | re.DOTALL | re.MULTILINE | re.VERBOSE,
)


def _strip_section_value(raw: str) -> str:
    """Нормализует значение секции, поддерживая разные варианты форматирования."""

    if not raw:
        return ""

    value = raw.lstrip()

    if value and value[0] in "=:–-":
        value = value[1:].lstrip()

    if not value:
        return ""

    if value.startswith("["):
        depth = 0
        end_index: Optional[int] = None
        for idx, ch in enumerate(value[1:], start=1):
            if ch == "[":
                depth += 1
            elif ch == "]":
                if depth == 0:
                    end_index = idx
                    break
                depth -= 1

        if end_index is not None:
            content = value[1:end_index].strip()
            trailing = value[end_index + 1 :].strip()
            if trailing:
                # если после закрывающей скобки остался текст — добавляем его
                content = f"{content} {trailing}" if content else trailing
            return content.strip()

        # закрывающая скобка не найдена — берём всё после первой откр. скобки
        value = value[1:]

    return value.strip()


def _extract_section(tag: str, text: str, required: bool = True) -> Optional[str]:
    tag_norm = tag.strip().upper()

    for match in _SECTION_BLOCK_RE.finditer(text or ""):
        if match.group("tag").strip().upper() != tag_norm:
            continue

        content = _strip_section_value(match.group("content") or "")
        return content

    if required:
        raise ValueError(f"Не удалось распарсить секцию [{tag}]")
    return None


def _normalize_item(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    # убираем типичные маркеры списков в начале строки
    cleaned = re.sub(r"^[\-–—•·\*\u2022]+", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    marker = cleaned.casefold().strip(" .,!?:;\"'()[]{}")
    if marker in _EMPTY_ITEM_MARKERS:
        return ""
    return cleaned


def _dedup_ordered(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for raw in items:
        item = _normalize_item(raw)
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _split(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[;\n]+", s)
    return _dedup_ordered(parts)


def _normalize_for_match(text: str) -> str:
    if not text:
        return ""
    lowered = text.casefold()
    cleaned = re.sub(r"[^0-9a-zа-яё]+", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


_PRODCLASS_NORMALIZED_CACHE: Dict[int, str] = {}


def _get_normalized_prodclass_name(prodclass_id: int) -> str:
    cached = _PRODCLASS_NORMALIZED_CACHE.get(prodclass_id)
    if cached is None:
        cached = _normalize_for_match(IB_PRODCLASS.get(prodclass_id, ""))
        _PRODCLASS_NORMALIZED_CACHE[prodclass_id] = cached
    return cached


def _parse_prodclass_id_from_digits(s: str) -> Optional[int]:
    if not s:
        return None
    for m in re.finditer(r"\d+", s):
        value = int(m.group(0))
        if value in IB_PRODCLASS:
            return value
    return None


def _guess_prodclass_by_name(section_text: str) -> Optional[int]:
    normalized = _normalize_for_match(section_text)
    if not normalized:
        return None

    best_id: Optional[int] = None
    best_score = 0.0
    for prodclass_id in sorted(IB_PRODCLASS):
        name_norm = _get_normalized_prodclass_name(prodclass_id)
        if not name_norm:
            continue
        if normalized == name_norm or normalized in name_norm or name_norm in normalized:
            return prodclass_id
        ratio = SequenceMatcher(None, normalized, name_norm).ratio()
        if ratio > best_score:
            best_score = ratio
            best_id = prodclass_id

    if best_id is not None and best_score >= 0.55:
        return best_id
    return None


async def _ensure_prodclass_name_vectors(
    embed_model: str,
) -> Tuple[List[int], List[List[float]]]:
    cached = _PRODCLASS_NAME_VECS_CACHE.get(embed_model)
    if cached is not None:
        return cached

    prodclass_items = sorted(IB_PRODCLASS.items())
    ids = [pid for pid, _ in prodclass_items]
    names = [name for _, name in prodclass_items]
    name_vecs = await _embeddings(names, embed_model)
    cached = (ids, name_vecs)
    _PRODCLASS_NAME_VECS_CACHE[embed_model] = cached
    return cached


async def _score_prodclass_by_text_vector(
    text_vector: Optional[List[float]], embed_model: str, prodclass_id: int
) -> Optional[float]:
    if text_vector is None:
        return None

    ids, name_vecs = await _ensure_prodclass_name_vectors(embed_model)
    try:
        idx = ids.index(prodclass_id)
    except ValueError:
        return None

    cos = _cosine_lists(text_vector, name_vecs[idx])
    return (cos + 1.0) / 2.0


async def _best_prodclass_by_text_vector(
    text_vector: Optional[List[float]], embed_model: str
) -> Tuple[Optional[int], Optional[float]]:
    if text_vector is None:
        return None, None

    ids, name_vecs = await _ensure_prodclass_name_vectors(embed_model)

    best_idx = -1
    best_cos = -1.0
    for idx, name_vec in enumerate(name_vecs):
        cos = _cosine_lists(text_vector, name_vec)
        if cos > best_cos:
            best_cos = cos
            best_idx = idx

    if best_idx < 0:
        return None, None

    score = (best_cos + 1.0) / 2.0
    if score < 0.3:
        return None, None

    return ids[best_idx], score


async def _guess_prodclass_by_embeddings(text_par: str, embed_model: str) -> _EmbeddingProdclassGuess:
    if not embed_model:
        return _EmbeddingProdclassGuess(text_vector=None, prodclass_id=None, score=None)

    payload = (text_par or "").strip()
    if not payload:
        return _EmbeddingProdclassGuess(text_vector=None, prodclass_id=None, score=None)

    truncated = payload[:6000]
    try:
        text_vec = await embed_single_text(truncated, embed_model)
    except Exception:
        return _EmbeddingProdclassGuess(text_vector=None, prodclass_id=None, score=None)

    if text_vec is None:
        return _EmbeddingProdclassGuess(text_vector=None, prodclass_id=None, score=None)

    try:
        best_id, best_score = await _best_prodclass_by_text_vector(text_vec, embed_model)
    except Exception:
        best_id, best_score = None, None

    return _EmbeddingProdclassGuess(text_vector=text_vec, prodclass_id=best_id, score=best_score)


async def _resolve_prodclass_id(
    section_text: str, text_par: str, embed_model: str
) -> _ProdclassResolution:
    embed_guess = await _guess_prodclass_by_embeddings(text_par, embed_model)

    prodclass_id = _parse_prodclass_id_from_digits(section_text)
    source = "model_reply"

    if prodclass_id is None:
        name_id = _guess_prodclass_by_name(section_text)
        if name_id is not None:
            prodclass_id = name_id
            source = "name_match"

    if prodclass_id is None:
        if embed_guess.prodclass_id is not None:
            return _ProdclassResolution(
                prodclass_id=embed_guess.prodclass_id,
                source="text_embedding_fallback",
                text_vector=embed_guess.text_vector,
                final_score=embed_guess.score,
                embed_best_id=embed_guess.prodclass_id,
                embed_best_score=embed_guess.score,
            )
        raise ValueError("В секции [PRODCLASS] не найдено целое число ID")

    final_score: Optional[float] = None

    if embed_model and embed_guess.text_vector is not None:
        candidate_score = await _score_prodclass_by_text_vector(
            embed_guess.text_vector, embed_model, prodclass_id
        )
        final_score = candidate_score

        if (
            embed_guess.prodclass_id is not None
            and embed_guess.score is not None
            and embed_guess.prodclass_id != prodclass_id
        ):
            candidate_score_norm = candidate_score or 0.0
            if embed_guess.score >= 0.55 and embed_guess.score >= candidate_score_norm + 0.1:
                prodclass_id = embed_guess.prodclass_id
                source = "text_embedding_override"
                final_score = embed_guess.score

    return _ProdclassResolution(
        prodclass_id=prodclass_id,
        source=source,
        text_vector=embed_guess.text_vector,
        final_score=final_score,
        embed_best_id=embed_guess.prodclass_id,
        embed_best_score=embed_guess.score,
    )


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

    resolution = await _resolve_prodclass_id(
        data["PRODCLASS_RAW"], text_par_for_fallback, embed_model
    )
    data["PRODCLASS"] = resolution.prodclass_id
    data["PRODCLASS_SOURCE"] = resolution.source
    if resolution.embed_best_id is not None:
        data["PRODCLASS_EMBED_GUESS"] = resolution.embed_best_id
    if resolution.embed_best_score is not None:
        data["PRODCLASS_EMBED_GUESS_SCORE"] = float(
            f"{resolution.embed_best_score:.2f}"
        )

    score = _parse_score(data["PRODCLASS_SCORE_RAW"])
    score_source = "model_reply"
    fallback_error: Optional[str] = None

    if resolution.source in {"text_embedding_override", "text_embedding_fallback"}:
        score = None
        score_source = resolution.source

    embedding_score_source: Optional[str] = None
    if resolution.final_score is not None:
        if resolution.source == "text_embedding_override":
            embedding_score_source = "text_embedding_override"
        elif resolution.source == "text_embedding_fallback":
            embedding_score_source = "text_embedding_fallback"
        else:
            embedding_score_source = "text_embedding_verify"

    if score is None:
        if resolution.final_score is not None:
            score = float(f"{resolution.final_score:.2f}")
            score_source = embedding_score_source or "text_embedding_verify"
        elif not embed_model:
            fallback_error = "embed_model is required to compute PRODCLASS_SCORE fallback"
        else:
            try:
                cls_name = IB_PRODCLASS.get(
                    resolution.prodclass_id, f"Класс {resolution.prodclass_id}"
                )
                truncated_text = text_par_for_fallback[:6000]
                vecs = await _embeddings([truncated_text, cls_name], embed_model)
                if len(vecs) < 2:
                    raise ValueError("embeddings response is incomplete")
                v1, v2 = vecs[0], vecs[1]
                sim = _cosine_lists(v1, v2)
                score = float(f"{(sim + 1.0) / 2.0:.2f}")
                score_source = "fallback_embeddings"
            except Exception as exc:
                fallback_error = str(exc)

    if score is None:
        score = 0.0
        score_source = "not_available"
    data["PRODCLASS_SCORE"] = score
    data["PRODCLASS_SCORE_SOURCE"] = score_source
    if fallback_error:
        data["PRODCLASS_SCORE_ERROR"] = fallback_error
    data["EQUIPMENT_LIST"] = _split(data["EQUIPMENT_RAW"])
    data["GOODS_LIST"] = _split(data["GOODS_RAW"])

    goods_type_list = _split(data["GOODS_TYPE_RAW"])
    goods_list = data["GOODS_LIST"]

    goods_type_source = "GOODS_TYPE"
    if not goods_type_list and goods_list:
        # Если LLM не заполнил GOODS_TYPE, используем GOODS как запасной вариант
        goods_type_list = list(goods_list)
        goods_type_source = "GOODS"

    data["GOODS_TYPE_LIST"] = goods_type_list
    data["GOODS_TYPE_SOURCE"] = goods_type_source
    return data


# ---------- matching to catalogs (с поддержкой вектора из БД) ----------

def _catalog_cache_get(embed_model: str, text: str) -> Optional[List[float]]:
    if not text:
        return None
    key = (embed_model, text)
    try:
        vec = _CATALOG_VECTOR_CACHE.pop(key)
    except KeyError:
        return None
    _CATALOG_VECTOR_CACHE[key] = vec
    return vec


def _catalog_cache_put(embed_model: str, text: str, vec: List[float]) -> None:
    if not text:
        return
    key = (embed_model, text)
    _CATALOG_VECTOR_CACHE[key] = vec
    _CATALOG_VECTOR_CACHE.move_to_end(key)
    while len(_CATALOG_VECTOR_CACHE) > _CATALOG_VECTOR_CACHE_LIMIT:
        _CATALOG_VECTOR_CACHE.popitem(last=False)


async def enrich_by_catalog(
    items: List[str],
    catalog: List[Dict[str, Any]],  # каждый: {id, name, vec?} где vec — str|(None)
    embed_model: str,
    min_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Возвращает [{text, match_id, score, vec, vec_str}]
    - эмбеддинг для item считаем всегда
    - для каталога: используем vec из БД, если есть, иначе считаем (с учётом LRU-кэша)
    - обработка каталога идёт батчами без хранения всех в памяти
    """
    if not items:
        return []

    cat_ids: List[int] = [int(c["id"]) for c in catalog]
    cat_names: List[str] = [str(c.get("name") or "").strip() for c in catalog]

    # векторы для items (их немного — оставляем в памяти до конца, т.к. они нужны в ответе)
    item_vecs = await _embeddings(items, embed_model)

    best_match_idx: List[int] = [-1] * len(items)
    best_match_cos: List[float] = [-1.0] * len(items)

    def _consider_catalog_vector(cat_index: int, cat_vec: Optional[List[float]]) -> None:
        if cat_vec is None:
            return
        for item_idx, item_vec in enumerate(item_vecs):
            cos = _cosine_lists(item_vec, cat_vec)
            if cos > best_match_cos[item_idx]:
                best_match_cos[item_idx] = cos
                best_match_idx[item_idx] = cat_index

    async def _process_pending(
        pending_idx: List[int],
        pending_names: List[str],
    ) -> None:
        if not pending_names:
            return
        embedded = await _embeddings(pending_names, embed_model)
        for local_pos, cat_vec in enumerate(embedded):
            idx = pending_idx[local_pos]
            name = pending_names[local_pos]
            _catalog_cache_put(embed_model, name, cat_vec)
            _consider_catalog_vector(idx, cat_vec)
        pending_idx.clear()
        pending_names.clear()

    pending_to_embed_idx: List[int] = []
    pending_to_embed_names: List[str] = []

    for idx, cat in enumerate(catalog):
        literal_vec = _parse_pgvector_text(str(cat.get("vec") or ""))
        if literal_vec is not None:
            _consider_catalog_vector(idx, literal_vec)
            continue

        name = cat_names[idx]
        cached = _catalog_cache_get(embed_model, name)
        if cached is not None:
            _consider_catalog_vector(idx, cached)
            continue

        if not name:
            continue

        pending_to_embed_idx.append(idx)
        pending_to_embed_names.append(name)
        if len(pending_to_embed_names) >= EMBED_BATCH_SIZE:
            await _process_pending(pending_to_embed_idx, pending_to_embed_names)

    await _process_pending(pending_to_embed_idx, pending_to_embed_names)

    out: List[Dict[str, Any]] = []
    for item_idx, text in enumerate(items):
        v_item = item_vecs[item_idx]
        best_idx = best_match_idx[item_idx]
        if best_idx < 0:
            out.append(
                {"text": text, "match_id": None, "score": None, "vec": v_item, "vec_str": _to_vec_str(v_item)}
            )
            continue

        best_cos = best_match_cos[item_idx]
        score = float(f"{(best_cos + 1.0) / 2.0:.2f}")
        match_id: Optional[int] = int(cat_ids[best_idx]) if score >= min_threshold else None

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

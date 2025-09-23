from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from openai import AsyncOpenAI
from app.config import settings
from app.models.ib_prodclass import IB_PRODCLASS

oai = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# --- настройки сопоставления ---
MATCH_THRESHOLD_EQUIPMENT = 0.45
MATCH_THRESHOLD_GOODS     = 0.45
EMBED_BATCH_SIZE          = 96


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
    all_vecs: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        chunk = texts[i:i + EMBED_BATCH_SIZE]
        resp = await oai.embeddings.create(model=embed_model, input=chunk)
        all_vecs.extend([d.embedding for d in resp.data])
    return all_vecs


def _cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A_norm @ B_norm.T


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
    if (s[0], s[-1]) in {('(', ')'), ('[', ']')}:
        s = s[1:-1]
    try:
        parts = s.split(",")
        return [float(p) for p in parts if p.strip() != ""]
    except Exception:
        return None


# ---------- parse model reply ----------

def _extract_section(tag: str, text: str, required: bool = True) -> str | None:
    m = re.search(rf"\[{re.escape(tag)}\]\s*=\s*\[(.*?)\]", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        if required:
            raise ValueError(f"Не удалось распарсить секцию [{tag}]")
        return None
    return m.group(1).strip()


def _split(s: str) -> list[str]:
    return [p.strip() for p in s.split(";") if p.strip()]


def _parse_prodclass_id(s: str) -> int:
    m = re.search(r"\d+", s)
    if not m:
        raise ValueError("В секции [PRODCLASS] не найдено целое число ID")
    return int(m.group(0))


def _parse_score(s: str | None) -> float | None:
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


async def parse_openai_answer(answer: str, text_par_for_fallback: str, embed_model: str) -> dict:
    data: Dict[str, Any] = {}
    data["DESCRIPTION"]         = _extract_section("DESCRIPTION", answer, required=True)
    data["PRODCLASS_RAW"]       = _extract_section("PRODCLASS", answer, required=True)
    data["PRODCLASS_SCORE_RAW"] = _extract_section("PRODCLASS_SCORE", answer, required=False)
    data["EQUIPMENT_RAW"]       = _extract_section("EQUIPMENT_SITE", answer, required=True)
    data["GOODS_RAW"]           = _extract_section("GOODS", answer, required=True)
    data["GOODS_TYPE_RAW"]      = _extract_section("GOODS_TYPE", answer, required=True)

    data["PRODCLASS"] = _parse_prodclass_id(data["PRODCLASS_RAW"])
    score = _parse_score(data["PRODCLASS_SCORE_RAW"])

    used_fallback = False
    if score is None:
        cls_name = IB_PRODCLASS.get(data["PRODCLASS"], f"Класс {data['PRODCLASS']}")
        v = await _embeddings([text_par_for_fallback[:6000], cls_name], embed_model)
        v1, v2 = np.array(v[0], dtype=np.float32), np.array(v[1], dtype=np.float32)
        sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
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
    catalog: List[dict],      # каждый: {id, name, vec?} где vec — str|(None)
    embed_model: str,
    min_threshold: float,
) -> List[dict]:
    """
    Возвращает [{text, match_id, score, vec, vec_str}]
    - эмбеддинг для item считаем всегда
    - для каталога: используем vec из БД, если есть, иначе считаем
    """
    if not items:
        return []

    # векторы для items
    item_vecs = await _embeddings(items, embed_model)
    A = np.array(item_vecs, dtype=np.float32)  # n x d

    # векторы каталога: пробуем взять из БД (vec), если нет — считаем
    cat_ids = [c["id"] for c in catalog]
    cat_names = [c["name"] for c in catalog]
    cat_vecs: List[List[float]] = []
    need_compute_texts: List[str] = []
    need_idx: List[int] = []
    for idx, c in enumerate(catalog):
        parsed = _parse_pgvector_text(c.get("vec") or "")
        if parsed is None:
            need_compute_texts.append(c["name"])
            need_idx.append(idx)
            cat_vecs.append([])  # placeholder
        else:
            cat_vecs.append(parsed)

    if need_compute_texts:
        computed = await _embeddings(need_compute_texts, embed_model)
        ptr = 0
        for j, idx in enumerate(need_idx):
            cat_vecs[idx] = computed[ptr]
            ptr += 1

    B = np.array(cat_vecs, dtype=np.float32) if cat_vecs else np.zeros((0, A.shape[1]), dtype=np.float32)
    sims = _cosine_matrix(A, B) if (A.size and B.size) else np.zeros((len(items), len(catalog)), dtype=np.float32)

    out: List[dict] = []
    for i, text in enumerate(items):
        if sims.shape[1] == 0:
            out.append({"text": text, "match_id": None, "score": None, "vec": item_vecs[i], "vec_str": _to_vec_str(item_vecs[i])})
            continue
        j = int(np.argmax(sims[i]))
        cos = float(sims[i, j])
        score = float(f"{(cos + 1.0) / 2.0:.2f}")
        match_id = int(cat_ids[j]) if score >= min_threshold else None
        out.append({
            "text": text,
            "match_id": match_id,
            "score": score if match_id is not None else None,
            "vec": item_vecs[i],
            "vec_str": _to_vec_str(item_vecs[i]),
        })
    return out

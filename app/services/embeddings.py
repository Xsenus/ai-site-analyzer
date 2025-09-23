# app/services/embeddings.py
from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional

import httpx

from app.config import settings

log = logging.getLogger("services.embeddings")

# ----------------------------
# Настройки из .env / Settings
# ----------------------------
# Ключ
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY") or (settings.OPENAI_API_KEY or "")

# Модель эмбеддингов: поддерживаем оба имени переменных окружения и значение из Settings
_EMBED_MODEL_ENV = os.getenv("EMBED_MODEL") or os.getenv("OPENAI_EMBED_MODEL")
OPENAI_EMBED_MODEL: str = _EMBED_MODEL_ENV or settings.embed_model or "text-embedding-3-large"

# Размерность вектора (для валидации при желании)
VECTOR_DIM: int = int(os.getenv("VECTOR_DIM") or settings.VECTOR_DIM or 3072)

# Внутренний провайдер (если есть)
INTERNAL_EMBED_URL: str = os.getenv("INTERNAL_EMBED_URL") or (settings.INTERNAL_EMBED_URL or "")
INTERNAL_EMBED_TIMEOUT: float = float(
    os.getenv("INTERNAL_EMBED_TIMEOUT") or settings.INTERNAL_EMBED_TIMEOUT or 6.0
)

# Доп. флаги/лимиты
EMBED_PREVIEW_CHARS: int = int(os.getenv("EMBED_PREVIEW_CHARS") or settings.EMBED_PREVIEW_CHARS or 160)
EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE") or settings.EMBED_BATCH_SIZE or 96)
EMBED_MAX_CHARS: int = int(os.getenv("EMBED_MAX_CHARS") or settings.EMBED_MAX_CHARS or 200_000)
DEBUG_OPENAI_LOG: bool = bool(
    (os.getenv("DEBUG_OPENAI_LOG") or "").lower() in {"1", "true", "yes", "on"} or settings.DEBUG_OPENAI_LOG
)


# ----------------------------
# Вспомогательные функции
# ----------------------------
def _chunk_text(t: str, max_chars: int) -> List[str]:
    """Нарезает длинный текст на куски max_chars, чтобы не упираться в лимиты."""
    if not t:
        return [""]
    if len(t) <= max_chars:
        return [t]
    return [t[i : i + max_chars] for i in range(0, len(t), max_chars)]


def _batches(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _avg_vectors(vectors: List[List[float]]) -> List[float]:
    """Простое усреднение списка векторов одинаковой длины (без numpy)."""
    if not vectors:
        return []
    acc = vectors[0][:]  # копия
    for vec in vectors[1:]:
        # страхуемся: длины могут разойтись для разных моделей; усредняем по минимальной длине
        m = min(len(acc), len(vec))
        for i in range(m):
            acc[i] += vec[i]
        if m < len(acc):
            # обрезаем до минимальной увиденной длины
            acc = acc[:m]
    n = len(vectors)
    if n > 1:
        for i in range(len(acc)):
            acc[i] /= n
    return acc


def _debug_preview(texts: List[str]) -> None:
    if not DEBUG_OPENAI_LOG:
        return
    for i, t in enumerate(texts):
        s = t[:EMBED_PREVIEW_CHARS].replace("\n", " ")
        log.info("EMBED[%03d] %s%s", i, s, "..." if len(t) > EMBED_PREVIEW_CHARS else "")


# ----------------------------
# Внутренний провайдер
# ----------------------------
async def _internal_embed(text: str, *, timeout: float) -> Optional[List[float]]:
    """
    Попытка обращения к вашему внутреннему сервису эмбеддингов.
    Если INTERNAL_EMBED_URL не задан — возвращает None.
    Ожидается JSON {"embedding": [...]}
    """
    if not INTERNAL_EMBED_URL:
        return None

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                INTERNAL_EMBED_URL,
                json={"text": text},
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
            data = r.json()
            vec = data.get("embedding")
            if isinstance(vec, list) and vec:
                return [float(x) for x in vec]
            log.warning("Internal embed: bad payload format")
            return None
    except Exception as ex:
        log.warning("Internal embed failed: %s", ex)
        return None


# ----------------------------
# OpenAI — одиночный текст
# ----------------------------
async def openai_embed(text: str, *, timeout: float = 12.0) -> List[float]:
    """
    Получить эмбеддинг для одного текста через OpenAI REST API.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    payload = {"model": OPENAI_EMBED_MODEL, "input": text}
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    if DEBUG_OPENAI_LOG:
        _debug_preview([text])

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        vec = data["data"][0]["embedding"]
        if not isinstance(vec, list) or not vec:
            raise RuntimeError("bad embedding payload from OpenAI")
        return [float(x) for x in vec]


# ----------------------------
# Публичные функции
# ----------------------------
async def make_embedding_or_none(text: str, *, timeout: float) -> Optional[List[float]]:
    """
    Сценарий 1: пробуем свой внутренний провайдер.
    Сценарий 2: если не получилось — пробуем OpenAI (если есть ключ).
    Сценарий 3: если ничего не вышло — None.
    """
    # 1) внутренний
    vec = await _internal_embed(text, timeout=min(timeout, INTERNAL_EMBED_TIMEOUT))
    if vec:
        return vec

    # 2) fallback OpenAI
    if OPENAI_API_KEY:
        try:
            # Режем на чанки и усредняем (на случай очень длинного текста)
            parts = _chunk_text(text, EMBED_MAX_CHARS)
            if len(parts) == 1:
                return await openai_embed(parts[0], timeout=timeout)

            # несколько чанков — эмбеддим и усредняем
            chunk_vectors: List[List[float]] = []
            for chunk in parts:
                chunk_vectors.append(await openai_embed(chunk, timeout=timeout))
            return _avg_vectors(chunk_vectors)
        except Exception as ex:
            log.warning("OpenAI fallback failed: %s", ex)

    # 3) не вышло
    return None


def validate_dim(vec: List[float], expect_dim: int) -> bool:
    """
    Если expect_dim > 0 — строго сверяем длину (иначе только что вектор непустой).
    """
    if not vec:
        return False
    if expect_dim and len(vec) != expect_dim:
        return False
    return True


# ----------------------------
# Дополнительно: пакетная обработка
# ----------------------------
async def embed_many(texts: List[str], *, timeout: float = 12.0) -> List[List[float]]:
    """
    Эмбеддинги для списка текстов.
    - Сначала пытаемся внутренним провайдером по одному тексту (если доступен).
    - Затем для оставшихся — OpenAI, батчами EMBED_BATCH_SIZE.
    - Длинные тексты режем на чанки EMBED_MAX_CHARS и усредняем.
    """
    results: List[Optional[List[float]]] = [None] * len(texts)

    # 1) Пытаемся внутренним провайдером
    if INTERNAL_EMBED_URL:
        for i, t in enumerate(texts):
            try:
                v = await _internal_embed(t, timeout=min(timeout, INTERNAL_EMBED_TIMEOUT))
                if v:
                    results[i] = v
            except Exception as ex:
                log.warning("Internal embed failed for item %d: %s", i, ex)

    # 2) OpenAI для тех, кто не обработан
    if OPENAI_API_KEY:
        # Разворачиваем тексты в чанки; для OpenAI выгодно отправлять пакетами
        to_process_indices: List[int] = [i for i, v in enumerate(results) if v is None]
        if to_process_indices:
            # Для батчинга подготовим пары (orig_idx, chunk_text)
            expanded_indices: List[int] = []
            expanded_texts: List[str] = []
            chunks_per_text: List[int] = []

            for idx in to_process_indices:
                parts = _chunk_text(texts[idx], EMBED_MAX_CHARS)
                chunks_per_text.append(len(parts))
                for p in parts:
                    expanded_indices.append(idx)
                    expanded_texts.append(p)

            if DEBUG_OPENAI_LOG:
                log.info(
                    "OpenAI embed_many: %d texts -> %d chunks (batch=%d, model=%s)",
                    len(to_process_indices),
                    len(expanded_texts),
                    EMBED_BATCH_SIZE,
                    OPENAI_EMBED_MODEL,
                )
                _debug_preview(expanded_texts[:min(len(expanded_texts), 10)])  # не спамим лог

            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            url = "https://api.openai.com/v1/embeddings"

            # собираем эмбеддинги чанков
            chunk_vectors: List[List[float]] = []
            async with httpx.AsyncClient(timeout=timeout) as client:
                for batch in _batches(expanded_texts, EMBED_BATCH_SIZE):
                    payload = {"model": OPENAI_EMBED_MODEL, "input": batch}
                    r = await client.post(url, headers=headers, json=payload)
                    r.raise_for_status()
                    data = r.json()
                    chunk_vectors.extend([item["embedding"] for item in data["data"]])

            # агрегируем обратно по исходным текстам (усреднение по чанкам)
            pos = 0
            for i, idx in enumerate(to_process_indices):
                k = chunks_per_text[i]
                if k == 1:
                    results[idx] = [float(x) for x in chunk_vectors[pos]]
                else:
                    vecs = [[float(x) for x in v] for v in chunk_vectors[pos : pos + k]]
                    results[idx] = _avg_vectors(vecs)
                pos += k

    # 3) финал: заменяем None на пустые векторы
    return [r if isinstance(r, list) else [] for r in results]

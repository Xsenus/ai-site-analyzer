import os
import pathlib
import sys

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import analyzer


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_enrich_by_catalog_prefers_existing_vectors(monkeypatch):
    analyzer._CATALOG_VECTOR_CACHE.clear()

    calls: list[list[str]] = []

    async def fake_embeddings(texts: list[str], embed_model: str) -> list[list[float]]:
        calls.append(list(texts))
        if texts == ["good1", "good2"]:
            return [[1.0, 0.0], [0.0, 1.0]], 0
        if texts == ["Catalog Two"]:
            return [[0.0, 1.0]], 0
        raise AssertionError(f"unexpected texts {texts}")

    monkeypatch.setattr(analyzer, "_embeddings", fake_embeddings)

    result = await analyzer.enrich_by_catalog(
        ["good1", "good2"],
        [
            {"id": 10, "name": "Catalog One", "vec": "[1.0,0.0]"},
            {"id": 20, "name": "Catalog Two", "vec": None},
        ],
        embed_model="fake-model",
        min_threshold=0.2,
    )

    assert calls == [["good1", "good2"], ["Catalog Two"]]
    assert result[0]["match_id"] == 10
    assert result[0]["score"] == 1.0
    assert result[1]["match_id"] == 20
    assert result[1]["score"] == 1.0
    assert result[0]["vec_str"].startswith("[")


@pytest.mark.anyio
async def test_enrich_by_catalog_uses_cache_for_catalog_embeddings(monkeypatch):
    analyzer._CATALOG_VECTOR_CACHE.clear()

    embed_calls: list[list[str]] = []

    async def fake_embeddings(texts: list[str], embed_model: str) -> list[list[float]]:
        embed_calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts], 0

    monkeypatch.setattr(analyzer, "_embeddings", fake_embeddings)

    catalog = [{"id": 5, "name": "Reusable", "vec": None}]
    items = ["query"]

    first = await analyzer.enrich_by_catalog(items, catalog, embed_model="fake", min_threshold=0.0)
    second = await analyzer.enrich_by_catalog(items, catalog, embed_model="fake", min_threshold=0.0)

    assert first[0]["match_id"] == 5
    assert second[0]["match_id"] == 5

    # Вызовы: [items], [catalog], [items]; без второго прохода по каталогу
    assert embed_calls == [["query"], ["Reusable"], ["query"]]

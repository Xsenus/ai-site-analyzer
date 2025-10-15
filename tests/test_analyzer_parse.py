import os
import pathlib
import sys

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import analyzer
from app.api import routes


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_parse_openai_answer_fallbacks_goods_type(monkeypatch):
    async def fake_embeddings(texts, embed_model):  # pragma: no cover - simple stub
        return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(analyzer, "_embeddings", fake_embeddings)

    answer = (
        "[DESCRIPTION]=[Описание]\n"
        "[PRODCLASS]=[41]\n"
        "[PRODCLASS_SCORE]=[]\n"
        "[EQUIPMENT_SITE]=[Станок]\n"
        "[GOODS]=[Товар А; Товар Б]\n"
        "[GOODS_TYPE]=[]"
    )

    parsed = await analyzer.parse_openai_answer(answer, "текст", "embed-model")

    assert parsed["PRODCLASS"] == 41
    assert parsed["PRODCLASS_SOURCE"] == "model_reply"
    assert parsed["PRODCLASS_SCORE_SOURCE"] == "fallback_embeddings"
    assert parsed["GOODS_TYPE_LIST"] == ["Товар А", "Товар Б"]
    assert parsed["GOODS_TYPE_SOURCE"] == "GOODS"


@pytest.mark.anyio
async def test_parse_openai_answer_merges_goods_type(monkeypatch):
    async def fake_embeddings(texts, embed_model):  # pragma: no cover - simple stub
        return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(analyzer, "_embeddings", fake_embeddings)

    answer = (
        "[DESCRIPTION]=[Описание]\n"
        "[PRODCLASS]=[41]\n"
        "[PRODCLASS_SCORE]=[0.75]\n"
        "[EQUIPMENT_SITE]=[Станок]\n"
        "[GOODS]=[Товар А; Товар Б]\n"
        "[GOODS_TYPE]=[Тип А; Товар Б]"
    )

    parsed = await analyzer.parse_openai_answer(answer, "текст", "embed-model")

    assert parsed["GOODS_TYPE_LIST"] == ["Тип А", "Товар Б", "Товар А"]
    assert parsed["GOODS_TYPE_SOURCE"] == "GOODS_TYPE"


@pytest.mark.anyio
async def test_parse_openai_answer_prodclass_source_name(monkeypatch):
    async def fail_embeddings(*_args, **_kwargs):  # pragma: no cover - sanity guard
        raise AssertionError("embeddings should not be called when score is present")

    monkeypatch.setattr(analyzer, "_embeddings", fail_embeddings)

    answer = (
        "[DESCRIPTION]=[Описание]\n"
        "[PRODCLASS]=[Предприятия механообработки (станочный парк и числовое программное управление)]\n"
        "[PRODCLASS_SCORE]=[0.52]\n"
        "[EQUIPMENT_SITE]=[Станок]\n"
        "[GOODS]=[Товар]\n"
        "[GOODS_TYPE]=[Тип]\n"
    )

    parsed = await analyzer.parse_openai_answer(answer, "текст", "embed-model")

    assert parsed["PRODCLASS"] == 41
    assert parsed["PRODCLASS_SOURCE"] == "name_match"


@pytest.mark.anyio
async def test_parse_openai_answer_requires_embed_model_for_score(monkeypatch):
    async def fail_embeddings(*_args, **_kwargs):  # pragma: no cover - should not run
        raise AssertionError("embeddings should not be invoked")

    monkeypatch.setattr(analyzer, "_embeddings", fail_embeddings)

    answer = (
        "[DESCRIPTION]=[Описание]\n"
        "[PRODCLASS]=[41]\n"
        "[PRODCLASS_SCORE]=[]\n"
        "[EQUIPMENT_SITE]=[Станок]\n"
        "[GOODS]=[Товар]\n"
        "[GOODS_TYPE]=[Тип]\n"
    )

    with pytest.raises(ValueError, match="embed_model"):
        await analyzer.parse_openai_answer(answer, "текст", "")


def test_ai_site_preview_formatting():
    enriched_items = [
        {
            "text": "Товар",
            "match_id": 7,
            "score": 0.85,
            "vec": [0.1, 0.2],
            "vec_str": "[0.1,0.2]",
        }
    ]

    preview = routes._ai_site_preview(
        enriched_items,
        pars_id=123,
        name_key="goods_type",
        id_key="goods_type_ID",
        score_key="goods_types_score",
    )

    assert preview == [
        {
            "text_par_id": 123,
            "goods_type": "Товар",
            "goods_type_ID": 7,
            "goods_types_score": 0.85,
            "text_vector": "[0.1,0.2]",
            "vector_dim": 2,
        }
    ]

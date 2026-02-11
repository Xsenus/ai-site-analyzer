import os
import pathlib
import sys

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import analyzer
from app.api.handlers import analyze_json as analyze_json_routes
from app.api.schemas import AnalyzeFromJsonRequest, CatalogItem, CatalogItemsPayload


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_parse_score_preserves_fractional_precision():
    assert analyzer._parse_score("0.815") == 0.815
    assert analyzer._parse_score("81.5") == 0.815


def test_normalize_score_uses_4_digits():
    assert analyzer._normalize_score(0.123456) == 0.1235


@pytest.mark.anyio
async def test_parse_openai_answer_fallbacks_goods_type(monkeypatch):
    async def fake_embeddings(texts, embed_model):  # pragma: no cover - simple stub
        return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(analyzer, "_embeddings", fake_embeddings)
    analyzer._PRODCLASS_NAME_VECS_CACHE.clear()

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
    assert parsed["PRODCLASS_SCORE_SOURCE"] == "text_embedding_verify"
    assert parsed["GOODS_TYPE_LIST"] == ["Товар А", "Товар Б"]
    assert parsed["GOODS_TYPE_SOURCE"] == "GOODS"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "description_line",
    [
        "[DESCRIPTION]=Профиль компании без квадратных скобок",
        "[DESCRIPTION]: [Профиль компании c альтернативным разделителем]",
        "[DESCRIPTION]=[Профиль компании без закрывающей скобки\nи дополнительной строкой",
    ],
)
async def test_parse_openai_answer_tolerates_description_format(monkeypatch, description_line):
    async def fake_embeddings(texts, embed_model):  # pragma: no cover - simple stub
        return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(analyzer, "_embeddings", fake_embeddings)

    answer = (
        f"{description_line}\n"
        "[PRODCLASS]=[41]\n"
        "[PRODCLASS_SCORE]=[0.80]\n"
        "[EQUIPMENT_SITE]=[Станок]\n"
        "[GOODS]=[Товар]\n"
        "[GOODS_TYPE]=[Тип]\n"
    )

    parsed = await analyzer.parse_openai_answer(answer, "текст", "embed-model")

    assert parsed["DESCRIPTION"].startswith("Профиль компании")
    assert parsed["PRODCLASS"] == 41


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

    assert parsed["GOODS_TYPE_LIST"] == ["Тип А", "Товар Б"]
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
async def test_parse_openai_answer_embedding_override(monkeypatch):
    async def fake_embeddings(texts, embed_model):  # pragma: no cover - deterministic vectors
        out = []
        for text in texts:
            marker = (text or "").casefold()
            if "механообработ" in marker:
                out.append([0.0, 1.0])
            else:
                out.append([1.0, 0.0])
        return out

    monkeypatch.setattr(analyzer, "_embeddings", fake_embeddings)
    analyzer._PRODCLASS_NAME_VECS_CACHE.clear()

    answer = (
        "[DESCRIPTION]=[Описание]\n"
        "[PRODCLASS]=[1]\n"
        "[PRODCLASS_SCORE]=[0.10]\n"
        "[EQUIPMENT_SITE]=[Станок]\n"
        "[GOODS]=[Товар]\n"
        "[GOODS_TYPE]=[Тип]"
    )

    site_text = "Компания выполняет механообработку деталей на станках с ЧПУ."

    parsed = await analyzer.parse_openai_answer(answer, site_text, "embed-model")

    assert parsed["PRODCLASS"] == 41
    assert parsed["PRODCLASS_SOURCE"] == "text_embedding_override"
    assert parsed["PRODCLASS_SCORE"] == 1.0
    assert parsed["PRODCLASS_SCORE_SOURCE"] == "text_embedding_override"
    assert parsed["PRODCLASS_EMBED_GUESS"] == 41
    assert parsed["PRODCLASS_EMBED_GUESS_SCORE"] == 1.0


@pytest.mark.anyio
async def test_parse_openai_answer_missing_embed_model_sets_default_score():
    answer = (
        "[DESCRIPTION]=[Описание]\n"
        "[PRODCLASS]=[41]\n"
        "[PRODCLASS_SCORE]=[]\n"
        "[EQUIPMENT_SITE]=[Станок]\n"
        "[GOODS]=[Товар]\n"
        "[GOODS_TYPE]=[Тип]\n"
    )

    parsed = await analyzer.parse_openai_answer(answer, "текст", "")

    assert parsed["PRODCLASS_SCORE"] == 0.0
    assert parsed["PRODCLASS_SCORE_SOURCE"] == "not_available"
    assert "PRODCLASS_SCORE_ERROR" in parsed


@pytest.mark.anyio
async def test_parse_openai_answer_fallback_failure_sets_default(monkeypatch):
    async def fail_embeddings(*_args, **_kwargs):  # pragma: no cover - stub failure
        raise RuntimeError("boom")

    monkeypatch.setattr(analyzer, "_embeddings", fail_embeddings)

    answer = (
        "[DESCRIPTION]=[Описание]\n"
        "[PRODCLASS]=[41]\n"
        "[PRODCLASS_SCORE]=[]\n"
        "[EQUIPMENT_SITE]=[Станок]\n"
        "[GOODS]=[Товар]\n"
        "[GOODS_TYPE]=[Тип]\n"
    )

    parsed = await analyzer.parse_openai_answer(answer, "текст", "embed")

    assert parsed["PRODCLASS_SCORE"] == 0.0
    assert parsed["PRODCLASS_SCORE_SOURCE"] == "not_available"
    assert parsed["PRODCLASS_SCORE_ERROR"].startswith("boom")


@pytest.mark.anyio
async def test_parse_openai_answer_filters_empty_equipment(monkeypatch):
    async def fake_embeddings(texts, embed_model):  # pragma: no cover - simple stub
        return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(analyzer, "_embeddings", fake_embeddings)

    answer = (
        "[DESCRIPTION]=[Описание]\n"
        "[PRODCLASS]=[41]\n"
        "[PRODCLASS_SCORE]=[0.75]\n"
        "[EQUIPMENT_SITE]=[нет; -; —; ;  ]\n"
        "[GOODS]=[Товар]\n"
        "[GOODS_TYPE]=[Тип]\n"
    )

    parsed = await analyzer.parse_openai_answer(answer, "текст", "embed-model")

    assert parsed["EQUIPMENT_LIST"] == []


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

    preview = analyze_json_routes._ai_site_preview(
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


def test_catalog_items_payload_accepts_object_with_vectors():
    payload = CatalogItemsPayload.model_validate(
        {
            "items": [
                {"id": 1, "name": "A", "vec": {"values": [0.1, 0.2]}},
                {"id": 2, "name": "B", "vec": {"literal": "[0.3,0.4]"}},
            ]
        }
    )

    normalized = analyze_json_routes._catalog_items_to_dict(payload)

    assert normalized == [
        {"id": 1, "name": "A", "vec": "[0.1000000,0.2000000]"},
        {"id": 2, "name": "B", "vec": "[0.3,0.4]"},
    ]


def test_catalog_items_to_dict_handles_plain_list():
    items = [
        CatalogItem(id=1, name="X", vec=[0.5, 0.6]),
        CatalogItem(id=2, name="Y", vec=None),
    ]

    normalized = analyze_json_routes._catalog_items_to_dict(items)

    assert normalized == [
        {"id": 1, "name": "X", "vec": "[0.5000000,0.6000000]"},
        {"id": 2, "name": "Y", "vec": None},
    ]


@pytest.mark.anyio
async def test_analyze_from_json_keeps_llm_answer_in_db_payload(monkeypatch):
    monkeypatch.setattr(analyze_json_routes, "build_prompt", lambda text: "PROMPT")

    async def fake_call_openai(prompt: str, model: str) -> str:
        assert prompt == "PROMPT"
        assert model == "fake-chat"
        return "LLM ANSWER"

    async def fake_parse(answer: str, text: str, embed_model: str) -> dict:
        assert answer == "LLM ANSWER"
        assert text == "source text"
        assert embed_model == "fake-embed"
        return {
            "DESCRIPTION": "Краткое описание",
            "PRODCLASS": 41,
            "PRODCLASS_SCORE": 0.77,
            "PRODCLASS_SOURCE": "model_reply",
            "PRODCLASS_SCORE_SOURCE": "model_reply",
            "EQUIPMENT_LIST": ["Станок"],
            "GOODS_LIST": ["Товар"],
            "GOODS_TYPE_LIST": ["Тип"],
            "GOODS_TYPE_SOURCE": "GOODS_TYPE",
        }

    async def fake_embed_single_text(text: str, embed_model: str) -> list[float]:
        assert text == "Краткое описание"
        assert embed_model == "fake-embed"
        return [0.1, 0.2]

    async def fake_enrich(
        items: list[str], catalog: list[dict], embed_model: str, threshold: float
    ) -> list[dict]:
        return [
            {
                "text": text,
                "match_id": 100 + idx,
                "score": 0.95,
                "vec": [0.2, 0.4],
                "vec_str": "[0.2,0.4]",
            }
            for idx, text in enumerate(items)
        ]

    async def fake_call_openai_with_usage(prompt: str, model: str):
        answer = await fake_call_openai(prompt, model)
        return answer, {"input_tokens": 200, "output_tokens": 50, "input_tokens_details": {"cached_tokens": 20}}

    async def fake_month_to_date_summary():
        class S:
            currency = "usd"
            period_start = 1
            period_end = 2
            spent_usd = 12.3
            limit_usd = 100.0
            prepaid_credits_usd = None
            remaining_usd = 87.7
        return S()

    monkeypatch.setattr(analyze_json_routes, "call_openai_with_usage", fake_call_openai_with_usage)
    monkeypatch.setattr(analyze_json_routes, "parse_openai_answer", fake_parse)
    monkeypatch.setattr(analyze_json_routes, "embed_single_text", fake_embed_single_text)
    monkeypatch.setattr(analyze_json_routes, "enrich_by_catalog", fake_enrich)
    monkeypatch.setattr(analyze_json_routes, "month_to_date_summary", fake_month_to_date_summary)

    request = AnalyzeFromJsonRequest(
        pars_id=10,
        text_par="source text",
        chat_model="fake-chat",
        embed_model="fake-embed",
        return_prompt=False,
        return_answer_raw=False,
    )

    response = await analyze_json_routes.analyze_from_json(request)

    assert response.answer_raw is None
    assert response.db_payload.llm_answer == "LLM ANSWER"
    assert response.parsed["LLM_ANSWER"] == "LLM ANSWER"
    assert response.db_payload.prodclass.id == 41
    assert response.db_payload.goods_types[0].vector.literal == "[0.2,0.4]"
    assert response.db_payload.equipment[0].vector.literal == "[0.2,0.4]"
    assert response.request_cost is not None
    assert response.request_cost.input_tokens == 200
    assert response.request_cost.cached_input_tokens == 20
    assert response.billing_summary is not None
    assert response.billing_summary.remaining_usd == 87.7


@pytest.mark.anyio
async def test_analyze_json_returns_502_on_parse_failure(monkeypatch):
    async def fake_llm(_prompt, _chat_model):
        return "[DESCRIPTION]=[ok]", {"input_tokens": 1, "output_tokens": 1}

    async def fail_parse(_answer, _text_par, _embed_model):
        raise ValueError("bad structure")

    monkeypatch.setattr(analyze_json_routes, "call_openai_with_usage", fake_llm)
    monkeypatch.setattr(analyze_json_routes, "parse_openai_answer", fail_parse)

    body = AnalyzeFromJsonRequest(text_par="sample", pars_id=1, company_id=1)

    with pytest.raises(analyze_json_routes.HTTPException) as exc_info:
        await analyze_json_routes.analyze_from_json(body)

    assert exc_info.value.status_code == 502
    assert str(exc_info.value.detail).startswith("LLM response parse failed:")

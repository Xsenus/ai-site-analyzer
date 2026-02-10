from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def _client() -> TestClient:
    return TestClient(app)


def test_site_available_prompt_success(monkeypatch):
    client = _client()

    async def fake_call_openai(prompt: str, model: str) -> str:
        assert "[DESCRIPTION_SCORE]" in prompt
        assert model == "gpt-4o"
        return (
            "[DESCRIPTION]=Ответ"
            "\n[DESCRIPTION_SCORE]=0.76"
            "\n[OKVED_SCORE]=0.81"
            "\n[PRODCLASS]=41"
            "\n[PRODCLASS_SCORE]=0.66"
            "\n[EQUIPMENT_SITE]=станок"
            "\n[GOODS]=детали"
            "\n[GOODS_TYPE]=код"
        )

    async def fake_parse(answer: str, text: str, embed_model: str) -> dict:
        assert "Ответ" in answer
        assert text.startswith("Компания производит")
        assert embed_model == settings.embed_model
        return {
            "DESCRIPTION": "Ответ",
            "DESCRIPTION_SCORE": 0.76,
            "OKVED_SCORE": 0.81,
            "PRODCLASS": 41,
        }

    from app.config import settings

    monkeypatch.setattr(
        "app.routers.prompt_templates.call_openai",
        fake_call_openai,
    )
    monkeypatch.setattr(
        "app.routers.prompt_templates.parse_openai_answer",
        fake_parse,
    )

    payload = {
        "text_par": "Компания производит детали и использует современное оборудование.",
        "company_name": "ООО \"Пример\"",
        "okved": "25.62",
    }

    response = client.post("/v1/prompts/site-available", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert isinstance(data["prompt"], str) and data["prompt"]
    assert data["prompt_len"] == len(data["prompt"])
    assert data["answer"].startswith("[DESCRIPTION]=")
    assert data["raw_response"] == data["answer"]
    assert data["answer_len"] == len(data["answer"])
    assert data["parsed"] == {
        "DESCRIPTION": "Ответ",
        "DESCRIPTION_SCORE": 0.76,
        "OKVED_SCORE": 0.81,
        "PRODCLASS": 41,
    }
    assert data["prodclass_by_okved"] is None
    assert data["duration_ms"] >= 0
    steps = [event["step"] for event in data["events"]]
    assert steps == [
        "validate_input",
        "build_prompt",
        "call_openai",
        "parse_answer",
    ]
    assert data["timings"]
    assert data["chat_model"] == "gpt-4o"
    assert data["embed_model"] == settings.embed_model
    assert data["error"] is None


def test_site_available_prompt_error(monkeypatch):
    client = _client()

    def boom(*_args, **_kwargs):  # pragma: no cover - вызывается в тесте
        raise RuntimeError("broken prompt")

    monkeypatch.setattr(
        "app.routers.prompt_templates.build_site_available_prompt",
        boom,
    )

    payload = {
        "text_par": "text",
        "company_name": "name",
        "okved": "01.11",
    }

    response = client.post("/v1/prompts/site-available", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is False
    assert data["prompt"] is None
    assert data["prompt_len"] == 0
    assert data["answer"] is None
    assert data["raw_response"] is None
    assert data["answer_len"] == 0
    assert any(event["status"] == "error" for event in data["events"])
    assert data["error"] == "broken prompt"


def test_site_unavailable_prompt_success(monkeypatch):
    client = _client()

    async def fake_call_openai(prompt: str, model: str) -> str:
        assert "PRODCLASS_by_OKVED" in prompt
        assert model == "gpt-4o"
        return "96"

    monkeypatch.setattr(
        "app.routers.prompt_templates.call_openai",
        fake_call_openai,
    )

    payload = {"okved": "25.62"}

    response = client.post("/v1/prompts/site-unavailable", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert isinstance(data["prompt"], str) and data["prompt"]
    assert data["prompt_len"] == len(data["prompt"])
    assert data["answer"] == "96"
    assert data["raw_response"] == "96"
    assert data["parsed"] == {"PRODCLASS": 96}
    assert data["prodclass_by_okved"] == 96
    assert data["timings"]
    assert data["chat_model"] == "gpt-4o"
    assert data["embed_model"] is None
    steps = [event["step"] for event in data["events"]]
    assert steps == [
        "validate_input",
        "build_prompt",
        "call_openai",
        "parse_answer",
    ]
    assert data["error"] is None


def test_site_unavailable_prompt_error(monkeypatch):
    client = _client()

    def boom(*_args, **_kwargs):
        raise ValueError("no okved")

    monkeypatch.setattr(
        "app.routers.prompt_templates.build_okved_prompt",
        boom,
    )

    response = client.post("/v1/prompts/site-unavailable", json={"okved": "00"})
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is False
    assert data["prompt"] is None
    assert data["prompt_len"] == 0
    assert data["raw_response"] is None
    assert any(event["status"] == "error" for event in data["events"])
    assert data["error"] == "no okved"

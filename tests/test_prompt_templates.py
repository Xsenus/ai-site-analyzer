from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def _client() -> TestClient:
    return TestClient(app)


def test_site_available_prompt_success():
    client = _client()
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
    assert data["duration_ms"] >= 0
    assert len(data["events"]) >= 2
    assert data["events"][0]["step"] == "validate_input"
    assert data["events"][1]["step"] == "build_prompt"
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
    assert any(event["status"] == "error" for event in data["events"])
    assert data["error"] == "broken prompt"


def test_site_unavailable_prompt_success():
    client = _client()
    payload = {"okved": "25.62"}

    response = client.post("/v1/prompts/site-unavailable", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert isinstance(data["prompt"], str) and data["prompt"]
    assert data["prompt_len"] == len(data["prompt"])
    assert data["duration_ms"] >= 0
    assert data["events"][0]["step"] == "validate_input"
    assert data["events"][1]["step"] == "build_prompt"
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
    assert any(event["status"] == "error" for event in data["events"])
    assert data["error"] == "no okved"

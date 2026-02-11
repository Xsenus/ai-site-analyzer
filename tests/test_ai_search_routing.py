import os
import pathlib
import sys

from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_API_KEY", "test-key")

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import create_app
from app.services import analyzer


def test_extract_section_and_normalize_values():
    payload = (
        "[DESCRIPTION] : [ Точное описание ]\n"
        "[EQUIPMENT_SITE]=[ -; Станок ЧПУ ; нет ; Линия ]\n"
    )

    description = analyzer._extract_section("DESCRIPTION", payload)
    equipment = analyzer._extract_section("EQUIPMENT_SITE", payload)

    assert description == "Точное описание"
    assert analyzer._split(equipment) == ["Станок ЧПУ", "Линия"]


def test_ai_search_api_path_and_alias(monkeypatch):
    async def fake_make_embedding_or_none(text: str, *, timeout: float):
        assert text
        assert timeout > 0
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr("app.routers.ai_search.make_embedding_or_none", fake_make_embedding_or_none)
    monkeypatch.setattr("app.routers.ai_search.validate_dim", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("app.main.make_embedding_or_none", fake_make_embedding_or_none)
    monkeypatch.setattr("app.main.validate_dim", lambda *_args, **_kwargs: True)

    app = create_app()
    client = TestClient(app)

    payload = {"q": "металлообработка"}

    api_resp = client.post("/api/ai-search", json=payload)
    alias_resp = client.post("/ai-search", json=payload)
    old_path_resp = client.post("/api/ai-search/ai-search", json=payload)

    assert api_resp.status_code == 200
    assert alias_resp.status_code == 200
    assert api_resp.json() == {"embedding": [0.1, 0.2, 0.3]}
    assert alias_resp.json() == {"embedding": [0.1, 0.2, 0.3]}
    assert old_path_resp.status_code == 404

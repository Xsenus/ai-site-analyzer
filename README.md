# AI Site Analyzer

AI Site Analyzer — stateless API-сервис на FastAPI для анализа контента сайтов,
генерации структурированных полей для downstream-систем и получения
эмбеддингов/профилей компании через OpenAI.

Сервис не хранит состояние между запросами: он получает входные данные,
выполняет inference/обогащение и возвращает результат в ответе.

## Что делает сервис

- Анализирует текст сайта (`/v1/analyze/json`) и формирует:
  - краткое описание компании;
  - предполагаемый производственный класс (`prodclass`);
  - списки товаров/услуг и оборудования;
  - сопоставления с входными каталогами (если каталоги переданы);
  - готовый `db_payload` для записи во внешнюю БД downstream-процессом.
- Возвращает эмбеддинги запросов (`/ai-search` и `/api/ai-search/ai-search`).
- Генерирует «длинный профиль компании + вектор» (`/v1/site-profile`).
- Позволяет отлаживать prompt-пайплайн отдельными роутами
  (`/v1/prompts/site-available`, `/v1/prompts/site-unavailable`).
- Показывает сводку расходов OpenAI за текущий месяц (`/v1/billing/remaining`).

---

## Архитектура и ключевые модули

```text
app/
├── main.py                    # сборка FastAPI-приложения, middleware, подключение роутов
├── run.py                     # production-runner uvicorn с воркерами
├── config.py                  # pydantic settings из .env
│
├── api/
│   ├── routes.py              # health + подключение analyze handler
│   ├── handlers/analyze_json.py
│   └── schemas.py             # запросы/ответы analyze + billing payload
│
├── routers/
│   ├── ai_search.py           # /api/ai-search/ai-search
│   ├── site_profile.py        # /v1/site-profile
│   ├── prompt_templates.py    # /v1/prompts/*
│   └── billing.py             # /v1/billing/remaining
│
├── services/
│   ├── analyzer.py            # prompt, вызов OpenAI, парсинг, матчинг каталогов
│   ├── embeddings.py          # внутренний embed провайдер + fallback на OpenAI
│   ├── site_profile.py        # генерация расширенного описания компании
│   ├── billing.py             # вызовы OpenAI Costs API
│   ├── pricing.py             # оценка стоимости запроса по usage
│   └── health.py              # ответ health check
│
├── schemas/                   # pydantic-схемы ai_search/site_profile/prompt_templates
├── models/ib_prodclass.py     # словарь классов производства
└── utils/                     # rate limiter и форматирование векторов
```

Подробная «карта выполнения» доступна в отдельном документе:
[docs/how_it_works.md](docs/how_it_works.md).

---

## Быстрый старт

### 1) Требования

- Python 3.11+
- доступ к OpenAI API (для большинства сценариев)

### 2) Установка

```bash
pip install -r requirements.txt
cp .env.example .env
```

### 3) Минимальная конфигурация

Достаточно задать:

```dotenv
OPENAI_API_KEY=...
CHAT_MODEL=gpt-4o
OPENAI_EMBED_MODEL=text-embedding-3-large
VECTOR_DIM=3072
```

### 4) Запуск

Для разработки:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8090 --reload
```

Для production-режима с настройками воркеров:

```bash
python -m app.run
```

После запуска:

- Swagger UI: `http://localhost:8090/docs`
- OpenAPI JSON: `http://localhost:8090/openapi.json`

---

## Эндпоинты

## 1) Health

### `GET /health`

Базовая проверка доступности процесса.

---

## 2) Основной анализ сайта

### `POST /v1/analyze/json`

Главный маршрут сервиса.

**На входе**: текст сайта + атрибуты компании + опциональные каталоги товаров,
оборудования и классов для матчинга.

**На выходе**:

- блок `parsed` с полями, извлечёнными из LLM-ответа;
- блоки `matches`/`counts`/`timings`;
- `db_payload` (структура для записи в downstream БД);
- `request_cost` и `billing_summary` (если включены billing-настройки).

Детальные контракты и интеграция:

- [docs/analyze_json_downstream_contract.md](docs/analyze_json_downstream_contract.md)
- [docs/analyze_json_integration.md](docs/analyze_json_integration.md)
- [docs/ai_analysis_flow_detailed.md](docs/ai_analysis_flow_detailed.md)

---

## 3) Отладка prompt-шаблонов

### `POST /v1/prompts/site-available`

Собирает prompt для кейса, когда есть текст сайта, вызывает модель и возвращает:
`prompt`, `answer`, `parsed`, события этапов и метрики времени.

### `POST /v1/prompts/site-unavailable`

Сценарий, когда сайта нет или текста недостаточно (опора на ОКВЭД/атрибуты).

Подробности:

- [docs/prompt_templates.md](docs/prompt_templates.md)

---

## 4) Site profile

### `POST /v1/site-profile`

Формирует расширенное фактологичное описание компании из `source_text`, затем
строит вектор этого описания.

**Результат**:

- `description` — готовый текст профиля;
- `description_vector` — эмбеддинг описания;
- `vector_dim` — размерность вектора;
- `prompt` — опционально (если `return_prompt=true`).

---

## 5) AI Search / эмбеддинги

Есть два варианта маршрута:

1. `POST /ai-search` — alias, возвращает только `{ "embedding": [...] }`.
2. `POST /api/ai-search/ai-search` — расширенный роутер с rate-limit и fallback
   форматом `{ "ids": ... }`, если вектор не получен.

Схемы и детали:

- [docs/ai_search.md](docs/ai_search.md)

---

## 6) Billing

### `GET /v1/billing/remaining`

Запрашивает MTD-расходы через OpenAI Costs API и возвращает:

- `spent_usd`
- `remaining_usd`
- `limit_usd` / `prepaid_credits_usd`
- `period_start`, `period_end`

Нужен `OPENAI_ADMIN_KEY`.

---

## Конфигурация (`.env`)

Ниже — ключевые переменные (полный пример в `.env.example`).

### OpenAI и модели

- `OPENAI_API_KEY` — ключ вызовов chat/embeddings.
- `CHAT_MODEL` — модель чата (по умолчанию `gpt-4o`).
- `OPENAI_EMBED_MODEL` (алиас: `EMBED_MODEL`) — embedding-модель.
- `VECTOR_DIM` — ожидаемая размерность вектора.

### Внутренний embedding-провайдер (опционально)

- `INTERNAL_EMBED_URL`
- `INTERNAL_EMBED_TIMEOUT`

### Runtime для ai-search

- `AI_SEARCH_TIMEOUT`
- `AI_SEARCH_RATE_LIMIT_PER_MIN`

### CORS и логирование

- `CORS_ALLOW_ORIGINS`
- `CORS_ALLOW_METHODS`
- `CORS_ALLOW_HEADERS`
- `CORS_ALLOW_CREDENTIALS`
- `LOG_LEVEL`

### Тюнинг эмбеддингов

- `EMBED_BATCH_SIZE`
- `EMBED_MAX_CHARS`
- `EMBED_PREVIEW_CHARS`
- `DEBUG_OPENAI_LOG`

### Billing

- `OPENAI_ADMIN_KEY`
- `BILLING_MONTHLY_LIMIT_USD`
- `BILLING_PREPAID_CREDITS_USD`
- `BILLING_COSTS_BASE_URL`

---

## Принципы работы и отказоустойчивости

- Stateless: сервис не пишет в БД.
- В `embeddings.py` сначала пытается внутренний провайдер, затем OpenAI fallback.
- Для длинных текстов эмбеддинги считаются по чанкам и усредняются.
- В `ai-search` используется sliding-window rate-limit по IP.
- В основном пайплайне `analyze/json` ответы LLM парсятся в структурированный вид,
  затем (если переданы каталоги) выполняется semantic matching по косинусу.
- Access-log middleware в `main.py` пишет единый формат логов и возвращает
  аккуратный `500` при необработанных исключениях.

---

## Тестирование

Запуск всех тестов:

```bash
pytest -q
```

В репозитории уже есть unit-тесты, в том числе на:

- pricing/billing расчёты;
- rate-limit;
- парсинг и обогащение analyze-пайплайна;
- prompt templates.

---

## Deployment

Для запуска через systemd используйте шаблон:

- [docs/systemd-service.md](docs/systemd-service.md)

Он рассчитан на запуск `python -m app.run` и управление параметрами через env.


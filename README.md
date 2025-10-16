# AI Site Analyzer

AI Site Analyzer — это асинхронный сервис на FastAPI, который анализирует сайты и
возвращает структурированный отчёт. Сервис работает поверх двух PostgreSQL
баз, обращается к OpenAI для генерации отчётов и эмбеддингов, а также
предоставляет вспомогательный эндпоинт `/api/ai-search` для получения
векторных представлений текстов.

## Основные возможности

- **Анализ сайтов без доступа к БД.** Основной эндпоинт `/v1/analyze/json`
  реализован в `app/api/handlers/analyze_json.py` и возвращает
  структурированный отчёт вместе с готовым `db_payload` для downstream-сервисов.
- **Поддержка двух баз данных.** Используются асинхронные движки SQLAlchemy с
  ленивой инициализацией и отдельными DSN для `postgres` и `parsing_data`.
- **Интеграция с OpenAI.** Модуль `app/services/analyzer.py` собирает промпт,
  вызывает LLM и разбирает ответ, а `app/services/embeddings.py` отвечает за
  получение эмбеддингов (внутренний сервис → OpenAI fallback).
- **Сервис эмбеддингов `/api/ai-search`.** Реализован в
  `app/routers/ai_search.py`, включает простой rate limit, нормализацию запросов
  и возможность подключить внутренний провайдер эмбеддингов.
- **Единая конфигурация через `.env`.** Настройки описаны в `app/config.py` и
  автоматически нормализуются (алиасы, значения по умолчанию, computed-поля).

## Структура проекта

```text
app/
├── api/                # публичные REST-роуты и pydantic-схемы
├── db/                 # создание движков, транзакции и пинг баз данных
├── models/             # справочники/модели домена
├── repositories/       # работа с БД (чтение текстов, поиск pars_id)
├── routers/            # дополнительные роутеры (ai-search)
├── schemas/            # pydantic-схемы для ai-search
├── services/           # интеграция с OpenAI, логика анализа
├── utils/              # вспомогательные утилиты (rate limiter и др.)
├── config.py           # pydantic Settings для загрузки окружения
├── logging_setup.py    # базовая настройка логирования
└── main.py             # точка входа FastAPI-приложения
```

## Подготовка окружения

1. **Python.** Требуется Python 3.11+.
2. **Зависимости.** Установите пакеты: `pip install -r requirements.txt`.
3. **Переменные окружения.** Скопируйте пример и укажите свои значения:
   ```bash
   cp .env.example .env
   ```
   Минимальный набор — DSN для PostgreSQL и ключ OpenAI. Полный список
   переменных приведён ниже.
4. **Запуск.** Используйте любой из вариантов:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8090 --reload
   # или
   python -m app.run
   ```

После запуска основное приложение доступно на `http://localhost:8090`.

## Ключевые переменные `.env`

| Переменная | Назначение |
| --- | --- |
| `POSTGRES_URL` | DSN основной БД (`postgresql+psycopg://…`). |
| `PARSING_URL` | DSN БД `parsing_data`. |
| `ECHO_SQL` | Логирование SQL (true/false). |
| `OPENAI_API_KEY` | Ключ OpenAI для генерации отчётов и эмбеддингов. |
| `CHAT_MODEL` | Модель диалога для анализа (по умолчанию `gpt-4o`). |
| `OPENAI_EMBED_MODEL` | Модель эмбеддингов (по умолчанию `text-embedding-3-large`). |
| `INTERNAL_EMBED_URL` | URL внутреннего сервиса эмбеддингов (опционально). |
| `AI_SEARCH_TIMEOUT` | Таймаут `POST /api/ai-search` в секундах. |
| `AI_SEARCH_RATE_LIMIT_PER_MIN` | Лимит запросов `/api/ai-search` в минуту. |
| `DEFAULT_WRITE_MODE` | Режим записи (`primary_only`, `dual_write`, `fallback_to_secondary`). |
| `CORS_ALLOW_*` | Настройки CORS (origins, methods, headers, credentials). |
| `LOG_LEVEL` | Уровень логирования FastAPI-приложения. |

Дополнительные параметры (например, `VECTOR_DIM`, `EMBED_BATCH_SIZE`,
`EMBED_MAX_CHARS`, `DEBUG_OPENAI_LOG`) можно оставить по умолчанию или
переопределить при необходимости.

## Проверка работоспособности

- `GET /health` — проверка подключения к базам.
- `POST /v1/analyze/json` — запуск анализа без прямого доступа к БД.
- `GET /api/ai-search/health` — health чек сервиса эмбеддингов.
- `POST /api/ai-search` — получение эмбеддинга или fallback-ответов.

## Работа с результатами анализа

Эндпоинт `POST /v1/analyze/json` не пишет данные в БД самостоятельно. Вместо
этого он возвращает детальный JSON с блоком `db_payload`, который повторяет
структуру таблиц (`ai_site_prodclass`, `ai_site_goods_types`, `ai_site_equipment`
и др.) и может быть напрямую использован downstream-сервисом. В ответе также
присутствуют `counts`, `timings`, предпросмотры записей и исходный ответ модели —
это помогает валидировать вставки и строить мониторинг.

Дополнительные детали контракта и примеры интеграции приведены в документации:

- [Контракт сервиса анализа с downstream-записью](docs/analyze_json_downstream_contract.md)
- [Рекомендации по интеграции `/v1/analyze/json`](docs/analyze_json_integration.md)

## Полезные советы

- Логирование уже настроено в `app/logging_setup.py`. При необходимости
  используйте переменную `LOG_LEVEL` или переопределите формат.
- В продакшене стоит подключить внешнее управление миграциями (Alembic) и
  секретами (Vault, AWS Secrets Manager и т.д.).
- Для локальной разработки можно настроить docker-compose с PostgreSQL и
  пробросить DSN через `.env`.

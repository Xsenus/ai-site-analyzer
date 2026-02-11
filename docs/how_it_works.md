# Как работает AI Site Analyzer (подробно)

Этот документ описывает фактический runtime-пайплайн сервиса: от входящего HTTP
запроса до финального JSON-ответа.

## 1. Запуск приложения

Точка входа — `app.main:create_app()`.

Что происходит при старте:

1. Создаётся экземпляр FastAPI.
2. Подключается CORS из переменных окружения.
3. Подключается middleware `AccessLogMiddleware`.
4. Подключаются роутеры:
   - базовые `api.routes` (включая `/health` и `/v1/analyze/json`);
   - `site_profile`, `prompt_templates`, `billing` (если импорт успешен);
   - `ai_search` под префиксом `/api`.
5. Добавляются alias-маршруты:
   - `POST /ai-search`
   - `GET /favicon.ico`

## 2. Глобальный middleware и логирование

`AccessLogMiddleware` для каждого запроса:

- определяет IP клиента (`x-forwarded-for` → `x-real-ip` → `request.client.host`);
- замеряет время обработки;
- логирует строку вида: `METHOD /path -> status (ms) ip=...`;
- при необработанной ошибке возвращает JSON `{"error":"internal error"}`.

Это даёт единый доступный access-log без зависимости от внешнего reverse proxy.

## 3. Пайплайн `/v1/analyze/json`

Маршрут: `app/api/handlers/analyze_json.py`.

Высокоуровнево:

1. Валидируется запрос (`AnalyzeFromJsonRequest`).
2. Нормализуются входные каталоги (товары/оборудование/prodclass).
3. Строится prompt через `services.analyzer.build_prompt(...)`.
4. Вызывается OpenAI (`call_openai_with_usage`).
5. Ответ модели парсится (`parse_openai_answer`) в структурированные поля.
6. Выполняется semantic enrichment (`enrich_by_catalog`) по входным каталогам.
7. Формируются `parsed`, `counts`, `timings`, `matches`, `db_payload`.
8. При наличии usage-токенов считается `request_cost` через pricing.
9. При наличии admin-key подтягивается `billing_summary` (MTD).
10. Возвращается финальный JSON без записи в БД.

### Что такое `db_payload`

`db_payload` — это «заготовка» для downstream-процесса записи в БД. Сервис
анализа сам запись не выполняет, только готовит структурированные данные.

## 4. Как строится prompt и парсится ответ LLM

Модуль: `app/services/analyzer.py`.

- Prompt включает:
  - правила формата ответа (`[DESCRIPTION]=...`, `[GOODS]=...` и т.д.);
  - справочник `IB_PRODCLASS`;
  - текст сайта.
- Вызов OpenAI делается через `AsyncOpenAI.chat.completions.create`.
- Модель запускается с `temperature=0.0`.
- Парсер ответа извлекает маркеры и превращает их в структуру Python.

Дополнительно в модуле реализованы:

- эмбеддинги (батчами);
- косинусная близость без numpy;
- кеширование векторов для часто встречающихся записей каталогов;
- fallback/переоценка класса производства по embedding-сходству.

## 5. Пайплайн эмбеддингов (`embeddings.py`)

Основная стратегия:

1. Пробуем внутренний embedding-сервис (`INTERNAL_EMBED_URL`).
2. Если не получилось — fallback в OpenAI embeddings API.
3. Если текст слишком длинный — режем на чанки (`EMBED_MAX_CHARS`) и усредняем
   векторы чанков.

Поддерживается пакетная функция `embed_many(...)`:

- сначала внутренний провайдер по элементам;
- затем OpenAI батчами `EMBED_BATCH_SIZE` для тех, кто не обработался.

## 6. Пайплайн `/api/ai-search/ai-search`

Маршрут: `app/routers/ai_search.py`.

Порядок:

1. Определение IP.
2. Rate-limit (`SlidingWindowRateLimiter`) по IP за минуту.
3. Нормализация `q`.
4. Попытка получить embedding.
5. Если embedding валиден по размерности — вернуть `{embedding:[...]}`.
6. Если embedding недоступен — вернуть fallback `{ids:{goods:[],equipment:[],prodclasses:[]}}`.

Есть alias `POST /ai-search` (в `main.py`) — он проще и возвращает только
embedding или 502.

## 7. Пайплайн `/v1/site-profile`

Маршрут: `app/routers/site_profile.py`, логика в `services/site_profile.py`.

Порядок:

1. Проверка `source_text`.
2. Сбор специального prompt-а на генерацию подробного профильного описания.
3. Вызов OpenAI chat.
4. Нормализация текста ответа (`_normalize_description`).
5. Построение embedding этого описания.
6. Возврат `description + description_vector`.

## 8. Пайплайн `/v1/prompts/*`

Роутер отладки prompt-шаблонов:

- `/v1/prompts/site-available`
- `/v1/prompts/site-unavailable`

Оба маршрута возвращают:

- итоговый prompt;
- raw answer модели;
- parsed структуру;
- список событий по шагам (`events`);
- тайминги по этапам (`timings`).

Это полезно для диагностики качества шаблонов и поведения модели.

## 9. Пайплайн billing

`/v1/billing/remaining` вызывает `services.billing.month_to_date_summary(...)` и
возвращает текущую MTD-сводку расходов.

Если не настроен admin-key или произошла ошибка провайдера, маршрут возвращает
HTTP 400/502 с объяснением.

## 10. Конфигурация и окружение

`app/config.py` использует `pydantic-settings` и поддерживает:

- загрузку из `.env`;
- алиасы переменных (например, `OPENAI_EMBED_MODEL` / `EMBED_MODEL`);
- дополнительные вычисляемые поля (например, `embed_model`).

Практически весь runtime контролируется переменными окружения, поэтому
переезд между dev/stage/prod делается без изменения кода.


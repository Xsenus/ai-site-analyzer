# Маршруты генерации промптов OpenAI

Этот документ описывает два REST‑маршрута, которые формируют готовые тексты запросов в OpenAI, вызывают модель и возвращают результат обработки вместе с телеметрией этапов. Они не пишут данные в базу и подходят для отладки шаблонов промптов отдельно от полного пайплайна `/v1/analyze/json`.

## Когда использовать

- Нужно проверить структуру и длину промпта перед встраиванием в основной сервис.
- Требуется быстро убедиться, что OpenAI корректно отвечает на шаблон (без
  матчинга каталогов и расчёта `db_payload`).
- Хотите собрать телеметрию по этапам (`validate_input`, `build_prompt`, `call_openai`,
  `parse_answer`) и посмотреть на промежуточные данные без обращения к БД.

## `POST /v1/prompts/site-available`

Запускает полный цикл анализа: формирует промпт по тексту сайта ("Запрос №1"), вызывает OpenAI и возвращает ответ модели вместе с разобранными данными.

### Тело запроса

| Поле | Тип | Обязательность | Описание |
|------|-----|----------------|----------|
| `text_par` | string | required | Текст, собранный с сайта компании. |
| `company_name` | string | required | Название компании, используется в секции `DESCRIPTION_SCORE`. |
| `okved` | string | required | Код ОКВЭД компании, используется в секции `OKVED_SCORE`. |
| `chat_model` | string | optional | Кастомная модель OpenAI. По умолчанию используется `settings.CHAT_MODEL`. |
| `embed_model` | string | optional | Модель эмбеддингов для постобработки. По умолчанию берётся из настроек (`settings.embed_model`). |

### Пример запроса

```json
POST /v1/prompts/site-available
{
  "text_par": "Компания производит ...",
  "company_name": "ООО \"Пример\"",
  "okved": "25.62"
}
```

### Ответ

```json
{
  "success": true,
  "prompt": "...",
  "prompt_len": 12345,
  "answer": "[DESCRIPTION]=...",
  "answer_len": 1300,
  "parsed": {
    "DESCRIPTION": "...",
    "DESCRIPTION_SCORE": 0.82,
    "OKVED_SCORE": 0.79,
    "PRODCLASS": 41,
    "PRODCLASS_SCORE": 0.71,
    "EQUIPMENT_LIST": ["Станок", "Лазерный комплекс"],
    "GOODS_LIST": ["Детали"],
    "GOODS_TYPE_LIST": ["Металлические изделия"],
    "GOODS_TYPE_SOURCE": "GOODS_TYPE"
  },
  "started_at": "2024-03-18T09:30:25.512000",
  "finished_at": "2024-03-18T09:30:26.100000",
  "duration_ms": 588,
  "events": [
    {
      "step": "validate_input",
      "status": "success",
      "detail": "Текст сайта и атрибуты компании получены",
      "duration_ms": null
    },
    {
      "step": "build_prompt",
      "status": "success",
      "detail": "Сформирован промпт длиной 12345 символов",
      "duration_ms": 112
    },
    {
      "step": "call_openai",
      "status": "success",
      "detail": "Получен ответ длиной 1300 символов",
      "duration_ms": 340
    },
    {
      "step": "parse_answer",
      "status": "success",
      "detail": "Ответ преобразован в структуру с 12 полями",
      "duration_ms": 92
    }
  ],
  "timings": {
    "build_prompt_ms": 112,
    "openai_ms": 340,
    "parse_ms": 92
  },
  "chat_model": "gpt-4o",
  "embed_model": "text-embedding-3-large",
  "error": null
}
```

* `success` — признак успешного завершения конвейера. При значении `false` поле `error` содержит текст ошибки, а `prompt`, `answer` и `parsed` могут отсутствовать.
* `prompt` — полный текст, который был отправлен в OpenAI. В шаблон включены разделы `DESCRIPTION`, `DESCRIPTION_SCORE`, `OKVED_SCORE`, `PRODCLASS`, `PRODCLASS_SCORE`, `EQUIPMENT_SITE`, `GOODS` и `GOODS_TYPE`, словарь `IB_PRODCLASS` с актуальными наименованиями классов и исходный текст сайта.
* `answer` — необработанный ответ модели OpenAI.
* `parsed` — результат постобработки: ключевые поля ответа с приведёнными оценками (`DESCRIPTION_SCORE`, `OKVED_SCORE`, `PRODCLASS`, списки оборудования и товаров, источник данных и т.д.).
* `timings` — словарь длительностей отдельных этапов (`build_prompt_ms`, `openai_ms`, `parse_ms`).
* `events` — хронологический список шагов с подробностями и длительностью.
* `chat_model` и `embed_model` — фактически использованные модели.
* `started_at`, `finished_at`, `duration_ms` — временные метки начала, окончания и суммарная длительность обработки.
* `error` — текст ошибки (если был сбой).

## `POST /v1/prompts/site-unavailable`

Формирует инструкцию для OpenAI только по ОКВЭД, вызывает модель и возвращает определённый класс производства.

### Тело запроса

| Поле | Тип | Обязательность | Описание |
|------|-----|----------------|----------|
| `okved` | string | required | Код ОКВЭД компании. |
| `chat_model` | string | optional | Модель OpenAI (по умолчанию `settings.CHAT_MODEL`). |

### Пример запроса

```json
POST /v1/prompts/site-unavailable
{
  "okved": "25.62"
}
```

### Ответ

```json
{
  "success": true,
  "prompt": "...",
  "prompt_len": 2150,
  "answer": "96",
  "answer_len": 2,
  "parsed": {
    "PRODCLASS": 96
  },
  "started_at": "2024-03-18T09:30:25.512000",
  "finished_at": "2024-03-18T09:30:25.600000",
  "duration_ms": 88,
  "events": [
    {
      "step": "validate_input",
      "status": "success",
      "detail": "Получен ОКВЭД компании",
      "duration_ms": null
    },
    {
      "step": "build_prompt",
      "status": "success",
      "detail": "Сформирован промпт длиной 2150 символов",
      "duration_ms": 23
    },
    {
      "step": "call_openai",
      "status": "success",
      "detail": "Получен ответ длиной 2 символа",
      "duration_ms": 49
    },
    {
      "step": "parse_answer",
      "status": "success",
      "detail": "Определён класс производства 96",
      "duration_ms": 8
    }
  ],
  "timings": {
    "build_prompt_ms": 23,
    "openai_ms": 49,
    "parse_ms": 8
  },
  "chat_model": "gpt-4o",
  "embed_model": null,
  "error": null
}
```

Поле `prompt` содержит инструкцию вернуть только одно число — идентификатор класса производства из справочника `IB_PRODCLASS`, который ближе всего к переданному ОКВЭД. Поле `parsed` содержит итоговый ID класса, а `timings`, `events` и `answer` позволяют проследить работу конвейера и диагностировать ошибки.

Оба маршрута доступны после развёртывания приложения FastAPI и автоматически добавляются в OpenAPI‑спецификацию (Swagger UI) в группе `analyze-json`, рядом с базовым эндпоинтом `/v1/analyze/json`.

## Практические советы

- Значения `chat_model` и `embed_model` берутся из переменных окружения, если не переданы в запросе. Убедитесь, что в `.env`
  выставлены актуальные модели, чтобы результаты совпадали с основным пайплайном.
- Поля `prompt` и `answer` могут содержать чувствительные данные; не логируйте их целиком в продакшене без маскировки.
- Используйте `events` и `timings` для мониторинга качества промптов: по длительности легко отследить задержки на стороне OpenAI
  и ошибки парсинга.

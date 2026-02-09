# Инструкция для сервиса-записи: как использовать `/v1/analyze/json`

Документ описывает полный цикл взаимодействия сервиса записи с API анализа сайтов.
Цель — получить от API готовую структуру `db_payload` и корректно синхронизировать
её с собственными таблицами БД. Сам сервис анализа больше не подключается к базам
данных: все необходимые данные должны приходить в запросе, а запись выполняется
downstream-системой.

> ℹ️ Для подробного описания формата ответа и входных данных смотрите также
> документ [Контракт сервиса анализа с downstream-записью](./analyze_json_downstream_contract.md).
> Логику подбора `PRODCLASS` и работу эмбеддингов описывает отдельный файл
> [Как сервис определяет класс производства](./prodclass_resolution.md).

## 1. Какие данные подготовить перед вызовом API

### 1.1. Основной текст сайта

* Источник: таблица `public.pars_site`.
* Необходимые поля: `id`, `text_par`, при наличии — `company_id` для логирования.
* Пример запроса:
  ```sql
  SELECT id, company_id, text_par
  FROM public.pars_site
  WHERE id = :pars_id;
  ```
* Если текст хранится в другом сервисе — сформируйте строку `text_par` вручную.
* Перед отправкой в API текст нужно очистить от управляющих символов и лишних
  пробелов (API делает `strip`, но лучше подготовить заранее).

### 1.2. Каталоги товаров и оборудования (опционально)

API принимает готовые каталоги в теле запроса и использует их для матчинга. Если
каталоги не передать, ответ вернёт только сырые списки из LLM без привязки к
ID.

* Источником могут быть ваши таблицы, например `public.ib_goods_types` и
  `public.ib_equipment`.
* Примеры запросов с дефолтными именами колонок:
  ```sql
  SELECT id, goods_type_name AS name, goods_type_vector::text AS vec
  FROM public.ib_goods_types
  ORDER BY id;
  
  SELECT id, equipment_name AS name, equipment_vector::text AS vec
  FROM public.ib_equipment
  ORDER BY id;
  ```
* Если вектор хранится массивом чисел или строкой pgvector (`[0.1, 0.2, ...]`),
  передайте значение как есть — сервис нормализует оба формата и поддерживает
  объекты `CatalogVector`.【F:app/api/handlers/analyze_json.py†L67-L127】

## 2. Как сформировать запрос к `/v1/analyze/json`

### 2.1. Метод и заголовки

* Метод: `POST`
* URL: `/v1/analyze/json`
* Заголовки: минимум `Content-Type: application/json`

### 2.2. Тело запроса

Схема описана моделью `AnalyzeFromJsonRequest`【F:app/api/schemas.py†L57-L66】.
Ключевые поля:

| Поле | Обязательность | Описание |
| --- | --- | --- |
| `text_par` | Да | Текст сайта для анализа. |
| `pars_id` | Нет | ID строки `pars_site` (используется в логах и далее в БД). |
| `company_id` | Нет | ID компании для расширенных логов. |
| `chat_model` | Нет | Название модели LLM. Если не указано, берётся из настроек (`settings.CHAT_MODEL`, по умолчанию `gpt-4o`).【F:app/api/handlers/analyze_json.py†L218-L228】【F:app/config.py†L10-L17】 |
| `embed_model` | Нет | Модель эмбеддингов. При отсутствии используется `settings.embed_model` (алиас к `OPENAI_EMBED_MODEL`).【F:app/api/handlers/analyze_json.py†L218-L228】【F:app/config.py†L12-L17】【F:app/config.py†L48-L51】 |
| `goods_catalog` | Нет | Каталог товаров — можно передать массив объектов `CatalogItem` либо объект `{ "items": [...] }`. |
| `equipment_catalog` | Нет | Каталог оборудования — формат тот же, что и для `goods_catalog`. |
| `return_prompt` / `return_answer_raw` | Нет | Флаги для отладки: вернуть ли сформированный промпт и «сырой» ответ LLM в верхнем уровне ответа. Даже если `return_answer_raw=false`, фактический текст сохраняется в `parsed.LLM_ANSWER` и `db_payload.llm_answer`. |

Пример минимального тела запроса:
```json
{
  "pars_id": 12345,
  "text_par": "ООО \"Пример\" изготавливает кабельные жгуты для автопрома..."
}
```

### 2.3. Полный пример запроса с каталогами

Ниже — референсный JSON, который гарантированно принимается `/v1/analyze/json`.
В нём заполнены все ключевые поля, добавлены каталоги и включён возврат
отладочной информации. При необходимости сократите каталоги до нескольких
позиций — сервис корректно обработает частичные списки.

```json
{
  "pars_id": 31,
  "company_id": 44,
  "text_par": "Главная | РУСЭНЕРГОСБЫТ\n\nЦентральный офис...",
  "chat_model": "gpt-4o",
  "embed_model": "text-embedding-3-large",
  "goods_catalog": {
    "items": [
      {
        "id": 101,
        "name": "Провода силовые",
        "vec": {
          "values": [0.12, -0.04, 0.33],
          "literal": "[0.12,-0.04,0.33]"
        }
      },
      {
        "id": 205,
        "name": "Трансформаторы",
        "vec": {
          "values": [0.08, 0.21, -0.17],
          "literal": "[0.08,0.21,-0.17]"
        }
      }
    ]
  },
  "equipment_catalog": {
    "items": [
      {
        "id": 11,
        "name": "Щиты распределительные",
        "vec": {
          "values": [0.44, -0.02, 0.19],
          "literal": "[0.44,-0.02,0.19]"
        }
      }
    ]
  },
  "return_prompt": true,
  "return_answer_raw": true
}
```

Что важно в этом примере:

1. **`text_par`** — обязательное поле. Здесь оно укорочено; реальный текст
   может быть длинным. Можно передать как обычную строку UTF‑8.
2. **`chat_model` и `embed_model`** заданы явно, чтобы убедиться, что вызывается
   нужная связка моделей. Если поля опустить, API возьмёт значения из
   конфигурации (см. раздел выше).
3. **Каталоги** можно передать как массив или как объект с полем `items`. Для
   каждого элемента `vec` принимает массив чисел, готовую строку pgvector или
   объект `{ "values": [...], "literal": "[...]" }`. Если `literal` не задан,
   сервис сам сформирует его по массиву значений.
4. **Флаги `return_prompt`/`return_answer_raw`** полезны на этапе интеграции:
   по ним легко сверить, что промпт и ответ модели выглядят ожидаемо. В боевом
   режиме их можно отключить.

Пример вызова с помощью `curl`:

```bash
curl -X POST "https://your-host/v1/analyze/json" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <TOKEN>" \
     -d @request.json
```

Где `request.json` — файл с телом из примера выше. Ответ придёт в формате
`AnalyzeFromJsonResponse` и будет содержать готовый блок `db_payload`.

### 2.4. Логирование на стороне сервиса

На своей стороне логируйте:
1. Исходный `pars_id`, длину текста и размер каталогов.
2. Итоговый URL и тело (без секрета API).
3. Код ответа и краткую сводку `counts`/`timings` после успешного вызова.

API само логирует весь внутренний пайплайн (построение промпта, вызов модели,
парсинг, вектора, обогащение каталогами).【F:app/api/handlers/analyze_json.py†L233-L398】

## 3. Что приходит в ответ и как его обработать

Ответ соответствует модели `AnalyzeFromJsonResponse`【F:app/api/schemas.py†L99-L114】.
Основной блок для записи в базу — `db_payload`【F:app/api/schemas.py†L90-L96】. Он содержит:

* `description` — итоговое описание компании.
* `description_vector` — вектор описания (если текст не пустой и эмбеддинг
  успешно посчитан).【F:app/api/handlers/analyze_json.py†L336-L358】
* `prodclass` — ID класса, скор, название, источник скора и способ определения ID.【F:app/api/handlers/analyze_json.py†L327-L334】
* `goods_types` — список товаров с привязкой к каталогу и векторами.
* `equipment` — аналогичный список оборудования.【F:app/api/handlers/analyze_json.py†L380-L409】
* `llm_answer` — исходный ответ модели (возвращается, если `return_answer_raw=true`).

Дополнительно полезны `counts`, `timings`, `catalogs` для мониторинга.

В словаре `parsed` теперь дублируются ключевые данные для аудита:

* `LLM_ANSWER` — оригинальный ответ модели (совпадает с `db_payload.llm_answer`).
* `AI_SITE_GOODS_TYPES` и `AI_SITE_EQUIPMENT` — предпросмотр строк для таблиц `ai_site_goods_types` и `ai_site_equipment` с расчётными полями.
  Эти структуры помогают сравнить будущие вставки с ожидаемыми данными из собственной логики записи.【F:app/api/handlers/analyze_json.py†L411-L440】
* `PRODCLASS_SOURCE` — способ определения класса производства. Значения: `model_reply`, `name_match`, `text_embedding_override`, `text_embedding_fallback`.
* `PRODCLASS_EMBED_GUESS`, `PRODCLASS_EMBED_GUESS_SCORE` — диагностика по эмбеддингам: какой класс подсказал текст сайта и с каким скором.
* `GOODS_TYPE_SOURCE` — источник списка товаров (`GOODS_TYPE` из ответа модели или резервный `GOODS`).

### 3.1. Обновление `pars_site`

1. Если `pars_id` известен, обновите поле `description` и при наличии — вектор
   `text_vector`.
   ```sql
   UPDATE public.pars_site
   SET description = :description,
       text_vector = CASE
           WHEN :description_vector_literal IS NULL THEN NULL
           ELSE CAST(:description_vector_literal AS vector)
       END
   WHERE id = :pars_id;
   ```
2. Рекомендуется выполнять в рамках одной транзакции вместе с остальными
   вставками. Если `pars_id` отсутствует, сохраните описание в своём хранилище
   без обновления таблицы.

### 3.2. Вставка в `ai_site_prodclass`

Используйте значения из `db_payload.prodclass`:
```sql
INSERT INTO public.ai_site_prodclass (text_pars_id, prodclass, prodclass_score)
VALUES (:pars_id, :prodclass_id, :prodclass_score);
```
Структура совпадает с блоком `db_payload.prodclass` в обработчике.【F:app/api/handlers/analyze_json.py†L325-L448】

### 3.3. Вставка в `ai_site_goods_types`

Для каждого элемента `db_payload.goods_types`:
```sql
INSERT INTO public.ai_site_goods_types
    (text_par_id, goods_type, goods_types_score, goods_type_ID, text_vector)
VALUES
    (:pars_id, :goods_name, :match_score, :match_id,
     CASE WHEN :vector_literal IS NULL THEN NULL ELSE CAST(:vector_literal AS vector) END);
```
Колонки совпадают с `db_payload.goods_types` в ответе API.【F:app/api/handlers/analyze_json.py†L381-L448】

### 3.4. Вставка в `ai_site_equipment`

Аналогично блокам оборудования:
```sql
INSERT INTO public.ai_site_equipment
    (text_pars_id, equipment, equipment_score, equipment_ID, text_vector)
VALUES
    (:pars_id, :equipment_name, :match_score, :match_id,
     CASE WHEN :vector_literal IS NULL THEN NULL ELSE CAST(:vector_literal AS vector) END);
```
Структура соответствует `db_payload.equipment` в ответе API.【F:app/api/handlers/analyze_json.py†L389-L448】

### 3.5. Обработка векторных данных

* `VectorPayload` содержит `values` (массив чисел) и `literal` (строка pgvector).【F:app/api/schemas.py†L69-L97】
* Если `literal` заполнен — можно использовать напрямую в `CAST(:literal AS vector)`.
* Если `literal` пуст, а `values` присутствуют, сформируйте строку `[...]` перед
  вставкой.

### 3.6. Финальные действия

1. После успешной записи зафиксируйте транзакцию (`COMMIT`).
2. В логах сохраните ID вставленных строк и ключевые показатели (`counts`,
   `timings`).
3. При необходимости прокиньте `parsed` или `answer_raw` в аудит.

## 4. Политика обработки ошибок и повторов

1. **HTTP-ошибка или таймаут при вызове API.**
   * Запишите ошибку в журнал.
   * Повторите попытку не более двух раз (всего максимум три вызова) с экспоненциальной
     паузой, чтобы избежать DDOS.
   * После трёх неудач — прекратите попытки, пометьте задачу как "failed", в БД ничего
     не записывайте.

2. **API вернул ошибку 4xx.**
   * Это ошибка входных данных. Не делайте повторов без изменения запроса.
   * Логируйте текст ошибки, перейдите к следующей задаче.

3. **Ошибка при разборе ответа или отсутствует `db_payload`.**
   * Такое возможно только при нарушении контракта. Зафиксируйте ответ целиком
     и не записывайте данные в БД.
   * Передайте событие в мониторинг/алертинг.

4. **Ошибки записи в БД.**
   * Работайте в транзакции: при исключении выполните `ROLLBACK` и не оставляйте
     частично записанных данных.
   * Повторите попытку записи максимум один раз (итого два коммита). Если ошибка
     повторяется — сохраните задачу в отдельную очередь для ручной проверки.

Следуя этим шагам, внешний сервис сможет безопасно использовать `/v1/analyze/json`
и синхронизировать данные анализа с собственной базой.

## 6. Новые поля стоимости и биллинга

`/v1/analyze/json` теперь возвращает два дополнительных поля верхнего уровня:

- `request_cost` — расчётная стоимость последнего запроса к LLM на основе `usage` и локальной таблицы цен модели.
  - `model`, `input_tokens`, `cached_input_tokens`, `output_tokens`, `cost_usd`.
- `billing_summary` — месячная сводка расходов из OpenAI Costs API (`/v1/organization/costs`), если настроен `OPENAI_ADMIN_KEY`.
  - `currency`, `period_start`, `period_end`, `spent_usd`, `remaining_usd`.

Важно: отсутствие `OPENAI_ADMIN_KEY` **не ломает** `/v1/analyze/json`; в этом случае `billing_summary` будет `null`.

Также добавлен отдельный эндпоинт:

- `GET /v1/billing/remaining?project_id=<optional>` — получить остаток лимита/кредитов за текущий месяц без запуска анализа сайта.

Переменные окружения:

- `OPENAI_ADMIN_KEY` — admin key организации OpenAI для доступа к org-level Costs API.
- `BILLING_MONTHLY_LIMIT_USD` — месячный лимит расходов.
- `BILLING_PREPAID_CREDITS_USD` — объём prepaid-кредитов (используется, если лимит не задан).
- `BILLING_COSTS_BASE_URL` — базовый URL billing API, по умолчанию `https://api.openai.com/v1`.


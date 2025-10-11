# Проверка и повторное использование сценария подбора оборудования

## 1. Как устроен текущий backend-роут

Маршрут `GET /v1/equipment-selection` вызывает функцию `compute_equipment_selection`, которая полностью реализует бизнес-логику.
Основные шаги:

1. Загружается карточка клиента и связанные с ним типы продукции и оборудования, найденные на сайте. Все числовые поля аккуратно приводятся к float, чтобы избежать проблем с Decimal/None. 【F:app/services/equipment_selection.py†L124-L187】
2. Собираются prodclass-результаты с сайта, агрегируются по среднему `prodclass_score` и сортируются по убыванию, чтобы сначала обрабатывать наиболее уверенные классы. 【F:app/services/equipment_selection.py†L188-L240】
3. Для каждого prodclass выполняется поиск производственных цехов и оборудования. При наличии прямых цехов оценка `SCORE_E1` вычисляется как `SCORE_1 × MAX(score, score_real)`. Если цехи не найдены, логика переключается на отраслевой фолбэк с коэффициентом 0.75. 【F:app/services/equipment_selection.py†L241-L381】
4. Максимальные результаты по `SCORE_E1` формируют таблицу `EQUIPMENT_1way`. 【F:app/services/equipment_selection.py†L386-L419】
5. Через связку goods_type → equipment собирается `SCORE_E2 = CRORE_2 × CRORE_3` и формируется список `EQUIPMENT_2way`. 【F:app/services/equipment_selection.py†L421-L483】
6. Оборудование, найденное напрямую на сайте, образует `EQUIPMENT_3way`, где `SCORE_E3 = equipment_score`. 【F:app/services/equipment_selection.py†L485-L519】
7. Все три списки объединяются в `EQUIPMENT_ALL` с приоритетом 1way > 2way > 3way при равных баллах, после чего таблицы пересоздаются в БД. 【F:app/services/equipment_selection.py†L521-L568】

## 2. Ручная проверка работы роута

1. Убедитесь, что у вас есть сетевой доступ к PostgreSQL по адресу `37.221.125.221:5464` и что хост доверяет вашему IP.
2. Выполните в корне репозитория:

   ```bash
   POSTGRES_URL='postgresql+psycopg://admin:9wR0ZJ3aAKC7@37.221.125.221:5464/postgres' \
   python - <<'PY'
   import asyncio, json
   from app.db.tx import get_primary_engine, run_on_engine
   from app.services.equipment_selection import compute_equipment_selection

   async def main():
       engine = get_primary_engine()
       if engine is None:
           raise RuntimeError('Engine not configured')
       async def action(conn):
           return await compute_equipment_selection(conn, 1)
       result = await run_on_engine(engine, action)
       print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))

   asyncio.run(main())
   PY
   ```

   Скрипт создаёт подключение, запускает расчёт и выводит полный JSON-ответ, совпадающий со структурой API.

3. Если требуется посмотреть таблицы в самой БД, можно подключиться к базе и выполнить `SELECT * FROM "EQUIPMENT_ALL" LIMIT 20;` после запуска скрипта.

> **Примечание.** В среде без доступа к внешней сети этот шаг завершится ошибкой `Network is unreachable`. Убедитесь, что доступ открыт.

## 3. Промпт для стороннего сервиса

Ниже пример промпта, который описывает бизнес-правила и формат реализации. Его можно использовать в другом сервисе (например, в задаче на исполнителя), чтобы воспроизвести расчёт на своей стороне.

```
Ты реализуешь расчёт трёх вариантов SCORE для оборудования клиента по данным PostgreSQL.
Используй следующий алгоритм (все ID относятся к БД parsing_data):

Вход: clients_requests.id = <CLIENT_ID>.

1. goods_types с сайта:
   - Таблица ai_site_goods_types → pars_site → clients_requests.
   - Для каждого goods_type_id возьми максимальный goods_types_score. Обозначь как CRORE_2.

2. Оборудование с сайта:
   - Таблица ai_site_equipment → pars_site → clients_requests.
   - Сохрани пары (equipment_id, equipment_score) для пути 3.

3. Prodclass:
   - Таблица ai_site_prodclass → pars_site → clients_requests, джойн с ib_prodclass ради имени.
   - Для каждого prodclass усредни prodclass_score → это SCORE_1. Обрабатывай prodclass в порядке убывания SCORE_1 и количества голосов.

4. Путь 1 (через prodclass):
   a) Найди цеха ib_workshops.prodclass_id = prodclass.
      - Если есть хотя бы один цех, собери оборудование ib_equipment.workshop_id ∈ найденные.
      - Для каждого оборудования посчитай SCORE_E1 = SCORE_1 × MAX(equipment_score, equipment_score_real).
   b) Если цехов нет, возьми industry_id из ib_prodclass.
      - Собери все prodclass той же отрасли и их workshops.
      - Если workshops найдены, рассчитай SCORE_E1 = SCORE_1 × 0.75 × MAX(equipment_score, equipment_score_real).
   c) Для каждого equipment.id сохрани максимальный SCORE_E1. Это таблица EQUIPMENT_1way.

5. Путь 2 (через goods_type):
   - Для каждого goods_type_id из шага 1 найди комбинации ib_equipment_goods → ib_goods → ib_goods_types.
   - Возьми equipment_score из ib_equipment (обозначь как CRORE_3).
   - SCORE_E2 = CRORE_2 × CRORE_3. Для каждого equipment.id оставь запись с максимальным SCORE_E2. Это EQUIPMENT_2way.

6. Путь 3 (через ai_site_equipment):
   - Для каждого equipment_id из шага 2 возьми SCORE_E3 = equipment_score.
   - Если найдётся имя в ib_equipment, используй его. Это EQUIPMENT_3way.

7. Итоговая таблица EQUIPMENT_ALL:
   - Объедини три списka. Для каждого equipment.id выбери запись с максимальным SCORE.
   - При равных SCORE приоритет 1way > 2way > 3way.
   - Храни поля (id, equipment_name, score, source).

Вывод: сериализуй четыре таблицы (1way, 2way, 3way, all) и вспомогательные данные (списки goods_types, prodclass_details и т.п.) в JSON.

Пример интерфейса на Python (async SQLAlchemy):

```python
async def compute_equipment_selection(conn: AsyncConnection, client_id: int) -> dict:
    # Повтори шаги 1–7, формируя структуры данных.
    return {
        "equipment_1way": [...],
        "equipment_2way": [...],
        "equipment_3way": [...],
        "equipment_all": [...],
        "goods_types": [...],
        "prodclass_details": [...],
    }
```
```

Промпт полностью описывает схему данных, формулы и приоритеты, что позволяет исполнителю воспроизвести алгоритм без доступа к этому репозиторию.

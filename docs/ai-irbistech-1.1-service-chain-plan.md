# AI IRBISTECH 1.1: ai-site-analyzer Service Chain Plan

## Контекст

После сквозной проверки продовой цепочки `library -> ai-integration -> ai-site-analyzer -> OpenAI`
`ai-site-analyzer` стал отдельным затронутым репозиторием. Раньше основное ТЗ было сосредоточено на
`ai-integration` и `library`, но реальная эксплуатация показала, что billing и health-контур analyzer
тоже критичны для стабильности всей связки.

## Текущий статус на 2026-04-23

- Блок rollout/monitoring для `ai-site-analyzer` реализован и выкачен на VPS `37.221.125.221`.
- Локальные проверки:
  - `python -m pytest` -> `55 passed`
  - `python -m compileall app` -> `success`
  - `bash -n deploy/ai-site-analyzer-rollout.sh deploy/install-ai-site-analyzer-systemd-units.sh` -> `success`
- Продовый rollout через `deploy/ai-site-analyzer-rollout.sh` проходит end-to-end:
  - installer раскладывает systemd unit-ы и monitoring env example;
  - `pytest` на сервере -> `55 passed`;
  - `compileall` на сервере -> `success`;
  - smoke на `/health` и `/v1/billing/remaining` проходит;
  - `python -m app.jobs.system_healthcheck --json` возвращает `severity=degraded`.
- На проде подтверждено:
  - `ai-site-analyzer.service` -> `active`
  - `ai-site-analyzer-healthcheck.timer` -> `active`
  - `system-health-state.json` создаётся в `/var/lib/ai-site-analyzer/`
  - JSON-артефакты пишутся в `/var/lib/ai-site-analyzer/system-health/`
- Во время rollout пойман и исправлен race-condition:
  timer больше не включается на installer-этапе внутри rollout, а стартует только в финальной фазе,
  чтобы не ловить ложный failed-status в окно остановки сервиса.

## Что уже сделано

- `GET /v1/billing/remaining` переведён в совместимый degraded-mode:
  при недоступном Costs API, отсутствии `OPENAI_ADMIN_KEY` или нехватке `api.usage.read`
  сервис возвращает `200` с `configured` и `error`, а не ломает downstream.
- Degraded billing summary теперь кэшируется, чтобы при повторных запросах не бить OpenAI Costs API
  на каждом обращении.
- Продовая сквозная проверка подтверждена на реальных вызовах:
  - `ai-site-analyzer /health` -> `200`
  - `ai-site-analyzer /v1/billing/remaining` -> `200`
  - `ai-integration /v1/analyze-service/health` -> `200`
  - `library /api/ai-analysis/billing` -> `200`

## Что закрывает этот блок

- Повторяемый rollout-helper для `ai-site-analyzer`.
- Standalone healthcheck job, который проверяет и liveness (`/health`), и billing-контур
  (`/v1/billing/remaining`).
- Systemd installer и единый monitoring env template.
- Документация по rollout, мониторингу и текущему degraded billing режиму.

## Реализация по шагам

### 1. Rollout и dev/runtime воспроизводимость

- Добавить `requirements-dev.txt` с `pytest`.
- Добавить `deploy/ai-site-analyzer-rollout.sh`.
- Скрипт должен поддерживать:
  - git и non-git layout;
  - загрузку `/etc/default/ai-site-analyzer-monitoring`, если файл существует;
  - установку `requirements.txt` и `requirements-dev.txt`;
  - запуск `pytest`;
  - `python -m compileall app`;
  - ожидание `GET /health`;
  - smoke на `GET /v1/billing/remaining`;
  - итоговый прогон `python -m app.jobs.system_healthcheck --json`.

### 2. Monitoring контур

- Добавить `app/jobs/system_healthcheck.py`.
- Job должен различать 3 состояния:
  - `ok`: `/health` отвечает `ok=true`, billing отдаёт валидный payload без `error`;
  - `degraded`: `/health` жив, billing отвечает `200`, но в payload есть `error`
    или `configured=false`;
  - `unhealthy`: `/health` недоступен/невалиден либо billing endpoint недоступен/ломается.
- Job должен поддерживать:
  - `state-file`;
  - `latest.json` и timestamped artifacts;
  - optional webhook alerts;
  - recovery alert по флагу.

### 3. Systemd installation

- Добавить:
  - `deploy/systemd/ai-site-analyzer-healthcheck.service`
  - `deploy/systemd/ai-site-analyzer-healthcheck.timer`
  - `deploy/systemd/ai-site-analyzer-monitoring.env.example`
  - `deploy/install-ai-site-analyzer-systemd-units.sh`
- Installer должен:
  - раскладывать unit-файлы;
  - устанавливать `.example` env template;
  - уметь bootstrap-ить `/etc/default/ai-site-analyzer-monitoring`, не перетирая существующий файл;
  - включать timer.

### 4. Тесты и документация

- Добавить автотесты на healthcheck summary/state/artifacts.
- Добавить автотесты, что env template покрывает все используемые
  `AI_SITE_ANALYZER_HEALTHCHECK_*` переменные.
- Обновить `README.md` и зафиксировать текущий статус сервиса.

## Что остаётся вне кода

- Для реального `spent_usd / remaining_usd` нужен `OPENAI_ADMIN_KEY` с scope `api.usage.read`.
- Пока такого scope у продового ключа нет, billing считается рабочим в degraded-mode, а не fully-green.
- Реальный webhook destination для alert-ов остаётся инфраструктурным хвостом и на код не влияет.

## Definition of Done

- Локально проходят `pytest` и `compileall`.
- На VPS rollout выполняется через `deploy/ai-site-analyzer-rollout.sh`.
- `ai-site-analyzer.service` и `ai-site-analyzer-healthcheck.timer` активны.
- `python -m app.jobs.system_healthcheck --json` на проде возвращает:
  - `severity=degraded`, если проблема только в OpenAI Costs scopes;
  - `severity=ok`, если billing полностью доступен;
  - `exit code != 0`, если реально сломан сервисный контур.

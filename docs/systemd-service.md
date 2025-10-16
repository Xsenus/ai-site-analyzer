# Пример unit-файла systemd

Ниже приведён рекомендуемый unit-файл для запуска сервиса под systemd.
Он использует виртуальное окружение и запускает `app.run`, чтобы работать с
настройками Uvicorn через переменные окружения.

```ini
[Unit]
Description=AI Site Analyzer (FastAPI + Uvicorn)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/ai-site-analyzer
Environment=PYTHONUNBUFFERED=1
Environment=HOST=0.0.0.0
Environment=PORT=8123
# Управляйте числом воркеров через переменную окружения,
# чтобы не редактировать ExecStart.
Environment=WORKERS=2
# UVICORN_RELOAD=1 оставляйте только для отладки, в проде держите 0.
Environment=UVICORN_RELOAD=0
ExecStart=/opt/ai-site-analyzer/.venv/bin/python -m app.run
Restart=on-failure
RestartSec=5
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

## Пояснения

- **ExecStart** использует `python -m app.run`, поэтому все параметры Uvicorn
  считываются из переменных окружения (`HOST`, `PORT`, `WORKERS`,
  `UVICORN_RELOAD`). Это упрощает обновления и гарантирует, что ограничения
  раннера (например, блокировка нескольких воркеров в режиме reload) сохраняются.
- **Переменные окружения** можно вынести в отдельный файл (`EnvironmentFile=`),
  если требуется хранить секреты отдельно.
- **Restart=on-failure** перезапускает процесс при падении. Если под нагрузкой
  процесс падает из-за OOM или нехватки дескрипторов, добавьте мониторинг на
  ресурсы и скорректируйте ulimit/MemoryMax.
- **WORKERS** подбирайте экспериментально. Как правило, `число_ядер * 2`
  достаточно для I/O-нагруженных задач, но начните с 2–4 и отслеживайте latency.
- **UVICORN_RELOAD** держите выключенным в продакшене — при его включении
  Uvicorn всё равно принудительно оставит только один воркер.

После добавления юнита выполните стандартные команды:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ai-site-analyzer.service
sudo systemctl status ai-site-analyzer.service
```

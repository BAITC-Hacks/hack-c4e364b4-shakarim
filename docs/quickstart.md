# Быстрый запуск TraceFlow

Команды выполняются в PowerShell из корня проекта. Требуется установленный Python 3.12.10.

## Подготовка

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
```

## Расчёт и интерфейс

```powershell
.\.venv\Scripts\python.exe src/pipeline.py --data data --output-dir output
.\.venv\Scripts\python.exe -m streamlit run ui/app.py
```

Откройте адрес, напечатанный Streamlit, обычно `http://localhost:8501`.
Выберите «Авто: output → mock» для просмотра рассчитанных результатов.

Исходные Parquet находятся в `data/`. Полный pipeline создаёт пять CSV в `output/`:

- `node_metrics.csv` — метрики узлов;
- `edges.csv` — направленные связи;
- `nodes_roles.csv` — роли и оценки;
- `clusters.csv` — кластеры;
- `top_nodes.csv` — очередь проверки.

## Проверка

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Подробности установки, сценарий для жюри и ограничения описаны в [README](../README.md).

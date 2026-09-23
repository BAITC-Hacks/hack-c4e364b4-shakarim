# Общий pipeline и передача CSV

В `main` объединены данные и метрики, аналитика и интерфейс. Общий pipeline
создаёт готовые файлы одного запуска и одного датасета в `output/`.
Streamlit читает эти файлы; для внешней папки используется `TRACEFLOW_RESULTS_DIR`.

| Ветка | Ответственность | Передаёт UI |
| --- | --- | --- |
| `dev/data-metrics` | Проверка исходных таблиц, граф и метрики | `node_metrics.csv`, `edges.csv` |
| `analytics` | Роли, приоритет, кластеры и объяснения | `nodes_roles.csv`, `clusters.csv`, `top_nodes.csv` |
| `dev/ui` | Отображение, поиск, направленные связи и демо | Сводка по GID; исходные CSV без изменения |

Таблица сохраняет разделение ответственности разработчиков. Для полного запуска
достаточно `main`; переключаться между рабочими ветками не требуется.

## Данные для интерфейса

`node_metrics.csv` содержит одну строку на GID. Полный контракт ветки метрик:

```text
gid,depth,is_seed,in_degree,out_degree,in_amount,out_amount,in_tx_count,out_tx_count,
unique_senders,unique_receivers,pass_through,pagerank,betweenness,truncated_by_depth
```

UI присоединяет по строковому GID только `depth`, `is_seed`, `truncated_by_depth`.
При отсутствии CSV допустимы метаданные из `data/nodes.parquet`.
Повторяющиеся GID в метаданных вызывают предупреждение; карточка остаётся доступной.
GID сохраняются строками и не преобразуются в JavaScript Number.

Для связей подходят `edges.csv` с `src,dst,sum_kzt,n_tx` или
`source,target,amount`. Можно передать исходный `edges.parquet` в `output/` или `data/`.
Стрелки UI всегда показывают исходное направление. Неориентированная проекция,
которую analytics использует для кластеризации, не заменяет исходные связи.

## Полный запуск из Parquet

Из корня репозитория после установки `requirements-test.txt`:

```powershell
python src/pipeline.py --data data --output-dir output
python -m streamlit run ui/app.py
```

Pipeline проверяет исходные данные, рассчитывает метрики и аналитику, сохраняет
пять файлов: `node_metrics.csv`, `edges.csv`, `nodes_roles.csv`, `clusters.csv`,
`top_nodes.csv`. В интерфейсе нажмите «Обновить данные» и выберите «Авто: output → mock».

## Повторный запуск аналитики по готовым метрикам

```powershell
python src/pipeline.py --input output/node_metrics.csv --edges output/edges.csv --output-dir output
```

Этот режим использует подготовленные метрики и обновляет три аналитических CSV.
Для кластеризации нужен `--edges`; без него каждый GID получает отдельный кластер.

## Передача результатов интерфейсу

Скопируйте пять CSV одного запуска в `output/` либо задайте отдельную папку:

```powershell
$env:TRACEFLOW_RESULTS_DIR = "C:\path\to\ready-output"
python -m streamlit run ui/app.py
```

При передаче только CSV история отдельных транзакций недоступна; для неё дополнительно
нужен `data/transactions.parquet` либо `TRACEFLOW_DATA_DIR` с исходными Parquet.
Это не мешает поиску, ролям и графу связей.

## Проверка общего контракта

```powershell
python -m unittest discover -s tests -v
```

Проверки включают полноту GID, соответствие сумм и количества транзакций,
метрики, роли, экспорт и открытие результатов в Streamlit. Для приложенного
датасета ожидаются 2 248 узлов, 3 119 направленных связей и 4 840 транзакций.
Обе оценки используют диапазон 0–1, `evidence` ограничен 200 символами.
У seed `pass_through` отсутствует, узлы `depth >= 4` без исходящих связей
не получают роль `terminal` по отсутствию исходящего потока.

Сгенерированные результаты оставлены локально в `output/` и исключены из Git.
Синтетический `mock/` версионируется и обеспечивает независимый запуск UI.

# Доказательный RAG по ТФКП

**Уденеев Александр Владимирович, группа Б05-602.**

Локальная система: BM25 и EmbeddingGemma через LanceDB, объединение
рангов RRF, соседние исходные блоки и два вызова Qwen3.5-4B.
Поиск и эмбеддинги выполняются на CPU, локальная генерация — на Apple GPU
через MLX. Проверяющие могут подставить свой OpenAI-совместимый endpoint.

## Результаты и материалы для сдачи

На 15 открытых примерах: 14/15 по наборам вариантов и отказам против
9/15 у первого baseline. Полный аудит обнаружил содержательные ошибки
и неполные цитаты, поэтому 14/15 не означает 14 доказательных ответов.
Опубликованные `dev_`-файлы относятся к открытому локальному пилоту.
Gold содержит ответы и страницы PDF, поэтому официальный F1 источников,
Q и E по нему не вычисляются. `runtime.json` код не читает: все настройки
находятся в `config.json`.

- [Итоговый отчёт](report.md), до 1500 слов.
- [Таблица пяти экспериментов](experiments.csv) и
  [анализ абляций](docs/comparison_report.md).
- [Аудит обоснований и ссылок](docs/citation_audit.md),
  [подробные записи](docs/citation_audit.jsonl).
- [8 авторских вопросов](own_questions.jsonl), [эталоны](own_gold.jsonl)
  и [пояснения к ним](docs/own_questions_notes.md).
- [Итоговые ответы](dev_answers.jsonl), [компактный trace](dev_trace.jsonl)
  и [происхождение файлов](submission_manifest.json).
- [Открытые вопросы](dev_questions.jsonl) и [эталоны](dev_gold.jsonl),
  воспроизводимые из `materials/tfkp_15_qa.md`.

## Запуск

Python 3.11. Команды выполняются из корня репозитория. `requirements.txt`
содержит RAG-клиент, `requirements-pdf.txt` — отдельно закреплённые
зависимости извлечения PDF без MLX. Корпус пересобирается из исходного
учебника; корпус, индекс и векторная БД исключены из Git.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-pdf.txt

python tools/prepare_data.py --pdf /path/to/book.pdf --output local_data --dev-output . --check-reference
python tools/download_embedder.py --config config.json
python build_index.py --corpus local_data/corpus_v2/corpus.jsonl --output local_data/index_hybrid --config config.json

export LLM_BASE_URL=http://127.0.0.1:8080/v1
export LLM_MODEL=/absolute/path/to/models/Qwen3.5-4B
export LLM_API_KEY=local
export SUBMISSION_ID=rag-hw1-local

python predict.py --index local_data/index_hybrid --questions dev_questions.jsonl --output local_data/run/answers.jsonl --trace local_data/run/trace.jsonl --config config.json
```

`prepare_data.py` создаёт первичный пилотный корпус, вопросы и gold,
декодирует глифы и строит окончательный `corpus.jsonl` с геометрией формул.
`--check-reference` проверяет совпадение корпуса и dev-файлов с данными
опубликованного прогона. Если готовый корпус уже предоставлен, подготовку
PDF можно пропустить и передать его путь в `build_index.py`.

28 октября 2026 подготовка данных проверена в отдельном окружении Python
3.11.15: корпус и оба dev-файла побайтно совпали с исходным прогоном.

Для внешнего endpoint укажите предоставленные URL, model ID и ключ.
При ответах по сети используется только этот endpoint. При индексации
генеративных вызовов нет. В trace учитываются все вызовы и usage;
повторных попыток нет. Лимиты задания проверяются перед каждым запросом.

[Полные команды](docs/reproduction.md) включают извлечение PDF,
подготовку пилота, локальный MLX-сервер, baseline, абляции и экспорт.

## Архитектура и ресурсы

Эмбеддер EmbeddingGemma-300M q8 ONNX, ревизия
`5090578d9565bb06545b4552f76e6bc2c93e4a66`; Qwen3.5-4B, ревизия
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Пути и параметры —
в `config.json`. Генерация в пилоте: temperature=0, thinking=false,
максимум 1024 выходных токена на вызов. Если курс задаёт иной runtime,
параметры нужно сверить перед повторным прогоном.

Mac M4 Pro, 48 GB RAM: сборка индекса 70,6 с, пиковая RSS сборщика
2,26 GB, индекс с энкодером 357 MB. Это не полная память процесса Qwen.

[Подробная архитектура](docs/rag_architecture.md) ·
[Извлечение формул](docs/extraction_notes.md) ·
[Исторический baseline](docs/baseline_report.md) ·
[Полная система](docs/hybrid_report.md).

## Структура

- В корне: CLI и модули RAG, пять профилей `config*.json`, зависимости,
  обязательные результаты, авторские вопросы и итоговый отчёт.
- `docs/`: дополнительные отчёты, аудит и воспроизведение.
- `materials/`: исходное условие и открытые примеры.
- `tools/`: подготовка данных, загрузчики, сравнения и экспорт.
- `local_data/`, `models/`, `index/`, `.venv/`: только локальные данные,
  исключены из Git. Исходный PDF хранится в `local_data/source/`.

Основные предсказания после аудита не исправлялись вручную. Компактный
dev trace сохраняет расходы и хеши запросов; полные сообщения с текстом
учебника остаются в игнорируемых локальных логах. Отдельные большие логи
остальных экспериментов не публикуются; результаты сохранены в CSV и отчёте.

## Использование ИИ

При решении домашнего задания использовался **OpenAI Codex**: для
проектирования и написания кода, подготовки и анализа экспериментов,
проверки источников, составления авторских вопросов и оформления отчёта.
Генерацию ответов в экспериментальных прогонах выполняла Qwen3.5-4B.
Подробности указаны в [отчёте](report.md#использование-ии).

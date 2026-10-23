# Воспроизведение экспериментов и извлечения

Все команды выполняются из корня репозитория.

**Уденеев Александр Владимирович, группа Б05-602.**

## Материалы для сдачи

- [Итоговый отчёт](../report.md), до 1500 слов.
- [Проверка обоснований и ссылок](citation_audit.md), подробные записи —
  `citation_audit.jsonl`.
- `own_questions.jsonl`, `own_gold.jsonl`: восемь авторских вопросов,
  [схема эталонов и пояснения](own_questions_notes.md).
- `dev_answers.jsonl`, `dev_trace.jsonl`: сохранённый итоговый пилот,
  [пять экспериментальных конфигураций](../experiments.csv).
- `submission_manifest.json`: происхождение и контрольные суммы файлов.

Официальный комплект не получен. Файлы с префиксом `dev_` содержат
15 локальных открытых примеров с локальными ID, а не официальный dev.
Точность 14/15 относится к вариантам и отказам; содержательный аудит
обнаружил ошибки даже при верных вариантах. Официальные Q/F1/E не
вычислены. Перед зачётной оценкой нужны официальный корпус, runtime
и повторный прогон. Публикация репозитория ещё не выполнялась.

Компактный trace сохраняет расходы всех вызовов и хеши запросов.
Полные запросы с текстом книги и исходные ответы модели остаются в
игнорируемом `local_data/`. Экспорт не меняет предсказания после аудита:

```bash
python tools/export_submission.py --run local_data/hybrid_canonical --questions local_data/pilot/pilot_questions.jsonl --corpus local_data/corpus_v2/corpus.jsonl
```

## Материалы

Условия находятся в `materials/homework_01_rag.md`, 15 открытых примеров — в
`materials/tfkp_15_qa.md`. Учебник PDF хранится локально в `local_data/source/`.

Официальные corpus/dev/gold/runtime, baseline и evaluator не получены.
Ниже описана подготовка локальных пилотных материалов.

В `local_data/pilot/` подготовлены отдельные предварительные материалы:

- `pilot_corpus.jsonl`: 147 блоков, по одному на страницу PDF; ID имеют
  префикс `pilot_`, поле `section` пока пустое.
- `pages.jsonl`: исходное извлечение, восстановленная кириллица и
  нормализованный текст каждой страницы.
- `extraction_report.json`: происхождение, SHA-256 PDF и ограничения
  извлечения. Часть формул теряет двумерную структуру; например, знак
  интеграла на странице 34 извлекается как `Z`.
- `pilot_questions.jsonl`: 15 вопросов в формате `id/question/options`.
- `pilot_gold_pages.jsonl`: эталоны из Markdown с номерами страниц PDF.
  Схема предварительная, допустимые официальные наборы `block_id` неизвестны;
  F1 ссылок по этому файлу оценивать нельзя.

Корпус извлекается программно без OCR, LLM и генеративного дополнения.
Официальные материалы при получении следует хранить отдельно от пилота.

## Основная система: EmbeddingGemma + LanceDB + BM25

Основной профиль теперь находится в `config.json`. Он объединяет
EmbeddingGemma-300M q8 и BM25 через RRF, добавляет соседние фрагменты
и использует два вызова Qwen: проверку вариантов и окончательный выбор.
Эмбеддинги и поиск работают на CPU; локальная Qwen — на Apple GPU через
MLX. LanceDB хранится в папке индекса, отдельный сервис не требуется.

[Архитектура](rag_architecture.md) · [Результаты и ошибки](hybrid_report.md).
На 15 пилотных вопросах получено 14/15 по наборам вариантов и отказам,
без технических/форматных ошибок после общей нормализации протокола.

```bash
python -m pip install -r requirements.txt
python tools/download_embedder.py --config config.json
python build_index.py --corpus local_data/corpus_v2/corpus.jsonl --output local_data/index_hybrid --config config.json

export LLM_BASE_URL=http://127.0.0.1:8080/v1
export LLM_MODEL=/absolute/path/to/models/Qwen3.5-4B
export LLM_API_KEY=local
export SUBMISSION_ID=rag-hw1-gemma-lance-hybrid-v1

python predict.py --index local_data/index_hybrid --questions local_data/pilot/pilot_questions.jsonl --output local_data/hybrid_run/answers.jsonl --trace local_data/hybrid_run/trace.jsonl --config config.json
python evaluate_pilot.py --questions local_data/pilot/pilot_questions.jsonl --gold local_data/pilot/pilot_gold_pages.jsonl --answers local_data/hybrid_run/answers.jsonl --trace local_data/hybrid_run/trace.jsonl --index local_data/index_hybrid --output local_data/hybrid_run/metrics.json
```

Сохранённый реальный прогон — `local_data/hybrid_v1/`. Итог общей проверки
тех же ответов после восстановления пропущенных названий полей —
`local_data/hybrid_canonical/`; дополнительных LLM-вызовов при этой
операции нет. Обновлённый `predict.py` выполняет ту же нормализацию сам.

```bash
python tools/revalidate_saved_run.py --questions local_data/pilot/pilot_questions.jsonl --index local_data/index_hybrid --trace local_data/hybrid_v1/trace.jsonl --output local_data/hybrid_canonical/answers.jsonl --output-trace local_data/hybrid_canonical/trace.jsonl
```

## Сравнение без контекста и две абляции

Три фиксированных профиля используют те же вопросы, модель, параметры
генерации, парсер ответа и токенные лимиты:

- `config.no_context.json`: один вызов Qwen без извлечения и текста книги.
  Разрешены собственные знания; ссылки всегда пусты. Оцениваются варианты
  и отказы, требование источников к этому контрольному варианту не применяется.
- `config.ablation_no_dense.json`: удалён только семантический канал.
  BM25, соседние блоки, упаковка контекста и оба вызова остаются.
- `config.ablation_one_pass.json`: полный гибридный поиск, но удалён первый
  вызов с таблицей доказательств. Источники упаковываются по прежнему
  бюджету, чтобы состав контекста совпадал с основной системой. Из промпта
  удалено требование сверять отсутствующую рабочую заметку.

После настройки endpoint из примера выше:

```bash
python tools/run_comparison.py --output local_data/comparison_v1 --index local_data/index_hybrid --questions local_data/pilot/pilot_questions.jsonl --gold local_data/pilot/pilot_gold_pages.jsonl
```

Запуски идут последовательно, без автоматических повторов. Уже существующую
папку результатов скрипт не перезаписывает. В подпапках `no_context`,
`no_dense`, `one_pass` находятся ответы, полный trace, метрики и разбор
каждого вопроса. `comparison_manifest.json` хранит контрольные суммы
исходного кода, конфигураций, вопросов, gold и manifest индекса. Gold
передаётся только оценщику после генерации, `predict.py` его не читает.

Результаты: без контекста 11/15, без dense 12/15, один вызов 10/15,
полная система 14/15. В новых прогонах ошибок формата и превышений
бюджета нет. Общая таблица — `experiments.csv`, анализ —
[comparison_report.md](comparison_report.md). Официальные Q/F1/E
пока не вычисляются.

## Первый RAG baseline

Baseline сохранён в `config.baseline.json`. `build_index.py` собирает
BM25 по всему корпусу на CPU. Один исходный блок
служит одним документом; текст формул и пространственные сетки сохраняются
целиком. Для поиска нормализуются Unicode, математические команды и русские
словоформы через общий префикс до семи букв. Заголовок раздела добавляется
к поисковому тексту дважды; выводимые источники остаются исходными.

`predict.py` делает один поиск по вопросу и всем вариантам, рассматривает
24 кандидата и набирает до 6 целых блоков в контекст. Размер полного
chat prompt измеряется токенизатором той же ревизии Qwen. При превышении
бюджета блок пропускается целиком. Один вызов модели формирует JSON;
проверяются ID вариантов, ссылки только на переданные блоки, длина
обоснования и согласованность отказа. Повторных вызовов и исправлений
ответа через LLM нет. Клиент допускает только две детерминированные
правки оформления: экранирование недопустимых JSON-последовательностей
в тексте и очистку ссылок при явном отказе с пустым списком вариантов.
Эти правки не выбирают варианты и сохраняются в `format_repairs` trace.
Технические сбои сохраняются отдельно в trace и
при пилотной оценке не засчитываются как правильные отказы.

Клиентская среда Python 3.11 переносима и отделена от локального
MLX-сервера: `requirements.txt` закрепляет зависимости индекса и клиента,
`requirements-local.txt` — полное окружение Mac для PDF и сервера.

```bash
python -m pip install -r requirements.txt
python build_index.py --corpus local_data/corpus_v2/corpus.jsonl --output index --config config.baseline.json

export LLM_BASE_URL=http://127.0.0.1:8080/v1
export LLM_MODEL=/absolute/path/to/models/Qwen3.5-4B
export LLM_API_KEY=local
export SUBMISSION_ID=rag-hw1-bm25-pilot-v1

python predict.py --index index --questions local_data/pilot/pilot_questions.jsonl --output local_data/baseline_run/answers.jsonl --trace local_data/baseline_run/trace.jsonl --config config.baseline.json
python evaluate_pilot.py --questions local_data/pilot/pilot_questions.jsonl --gold local_data/pilot/pilot_gold_pages.jsonl --answers local_data/baseline_run/answers.jsonl --trace local_data/baseline_run/trace.jsonl --index index --output local_data/baseline_run/metrics.json
```

Переменные окружения задают заменяемый endpoint и модель; заголовки
`X-Question-ID` и `X-Submission-ID` передаются в каждом запросе. В trace
сохраняются запрос без ключа, ответ, ID источников, фактические usage,
контрольные суммы и время. Предсказание читает токенизатор только из
индекса и обращается по сети только к endpoint. На сборке индекс включает
файлы токенизатора из локальной модели; при отсутствии локальной копии
скачивается только токенизатор фиксированной ревизии из Hugging Face.

Профиль baseline **локальный и предварительный**: temperature=0,
thinking выключен, максимум 1024 выходных токена, вход до 5600 токенов,
тайм-аут 180 секунд. Это не выданный baseline и не официальный runtime.
Параметры нельзя считать зачётными до получения `runtime.json`.
Оценщик выводит точность набора вариантов/отказа и диагностическое покрытие
страниц. Допустимые наборы официальных блоков неизвестны, поэтому F1
источников, Q, E и итоговый балл остаются неопределёнными.

Реальный прогон 23 октября 2026 сохранён в `local_data/baseline_v1/`.
После двух правок оформления те же ответы повторно обработаны без
генерации в `local_data/baseline_canonical/`: 9/15 по наборам вариантов и
отказам, 4 технических/форматных отказа, 2 неверных набора вариантов.
Разбор, ресурсы и ограничения — в [baseline_report.md](baseline_report.md).
Для воспроизведения именно обработки сохранённого прогона:

```bash
python tools/revalidate_saved_run.py --questions local_data/pilot/pilot_questions.jsonl --index index --trace local_data/baseline_v1/trace.jsonl --output local_data/baseline_canonical/answers.jsonl --output-trace local_data/baseline_canonical/trace.jsonl
```

## Извлечение с учётом шрифтов и геометрии

Для улучшенной версии используются исходные коды глифов, встроенные
`ToUnicode` и Type 1 `Encoding`, контуры математических символов и векторные
линии PDF. Для кириллических Type 3 шрифтов этого учебника явно выбирается
профиль CM-Super T2A. Такой профиль не определяется по диапазону байтов
автоматически и не должен применяться к произвольным PDF без проверки.

Сравнение подходов, выбранный способ и границы визуальной проверки описаны
в [extraction_notes.md](extraction_notes.md). Однозначные связи формул
представлены читаемым текстом с математической разметкой. Неоднозначные
области помечены и представлены моноширинной сеткой исходных глифов и черт;
пробелы и переводы строк в этих областях нужно сохранять при индексации.
Сетку нельзя считать семантически восстановленным LaTeX.

```bash
.venv/bin/python tools/pdf_font_decoder.py --pdf /path/to/book.pdf --type3-encoding cm-super-t2a --output-dir local_data/font_analysis
.venv/bin/python tools/build_pdf_corpus.py --pdf /path/to/book.pdf --glyphs local_data/font_analysis/glyph_pages.jsonl --old-corpus local_data/pilot/pilot_corpus.jsonl --output local_data/corpus_v2
```

Результаты находятся в `local_data/corpus_v2/`:

- `corpus.jsonl` — блоки с полями `block_id,page,section,text`; ID с префиксом
  `pdfv2_` являются локальными. Разделы берутся из оглавления исходного PDF.
- `provenance.jsonl` — происхождение блоков, страницы и геометрия строк.
- `source_geometry.jsonl` — глифы, их исходные коды, шрифты и координаты.
- `layout_diagnostics.jsonl` — распознанные пространственные связи и
  неоднозначности; отсутствие неизвестных глифов не означает правильности
  структуры каждой формулы.
- `quality_report.json` — статус сборки, покрытие, ограничения и диагностика.
  Статус черновика нужно учитывать до использования корпуса в экспериментах.

Независимая сверка конкретной сборки сохраняется в
`local_data/extraction_comparison/final_audit.md` и `final_audit.json`;
при сопоставлении нужно учитывать записанный там SHA-256 корпуса.

Обычные экстракторы сравниваются отдельно:

```bash
.venv/bin/python tools/compare_pdf_extractors.py --help
```

Их полные выгрузки и визуальное сравнение сохранены в
`local_data/extraction_comparison/`. Извлечение не использует OCR,
генеративные модели или внешние сервисы; исходный PDF сохраняется.

## Подготовка пилотных данных

Python 3.11; извлечение PDF требует `pypdf==6.10.0`.

```bash
python tools/prepare_pilot_corpus.py --pdf /path/to/book.pdf --output local_data/pilot
python tools/prepare_pilot_questions.py --source materials/tfkp_15_qa.md --output local_data/pilot
```

## Локальная Qwen

Исходные веса: `Qwen/Qwen3.5-4B`, без дополнительной квантизации.
Каталог `models/Qwen3.5-4B/` исключён из Git вместе с локальными данными,
окружением и кэшами. Ревизия модели:
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

Скрипт загрузки сохраняет точную ревизию в `download_manifest.json` рядом
с весами. Для повторного скачивания требуется `huggingface_hub`:

```bash
python tools/download_qwen.py --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

Для Apple Silicon используется `mlx-lm`: он загружает языковую часть
оригинального чекпойнта. Полные версии локальных зависимостей сохранены
в `requirements-local.txt`; этот набор предназначен для Mac с Apple Silicon.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-local.txt
```

Запуск из корня проекта:

```bash
.venv/bin/python tools/serve_qwen.py
```

Загрузчик сервера работает офлайн с весами из `models/`. На 23 октября 2026
веса полностью скачаны, сервер запущен на M4 Pro 48 GB; короткий запрос
вернул «Готово» и статистику: 19 входных и 4 выходных токена. API подтвердил
использование GPU Apple. Параметры этого короткого запроса служили только
для проверки развёртывания. Позднее выполнен пилотный прогон 15 вопросов;
результаты приведены в разделе baseline выше.

Сервис слушает `http://127.0.0.1:8080/v1`; Chat Completions доступен по
`/v1/chat/completions`. Для клиента RAG:

```bash
export LLM_BASE_URL=http://127.0.0.1:8080/v1
export LLM_MODEL="$(pwd)/models/Qwen3.5-4B"
export LLM_API_KEY=local
```

Локальный запуск предназначен для разработки. Официальные настройки
sampling, thinking, тайм-аутов и ограничений памяти пока неизвестны:
`runtime.json` не получен.

Документация: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B),
[MLX-LM server](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md).

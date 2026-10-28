# Полное воспроизведение из PDF

Все команды выполняются из корня репозитория под Python 3.11. Исходник —
учебник Горяйнова и Половинкина, 2025, ссылка на PDF находится в
[условии](../materials/homework_01_rag.md). Полученный файл передаётся
явным путём; имя файла и текущая папка запуска не влияют на текст корпуса.

## 1. Окружение и зависимости

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-pdf.txt
```

`requirements-pdf.txt` фиксирует зависимости извлечения и их транзитивные
пакеты. Для подготовки корпуса модели, PyTorch и MLX не нужны.
Для индексации и предсказания дополнительно установите RAG-клиент:

```bash
python -m pip install -r requirements.txt
```

`requirements-local.txt` сохраняет полное исходное Mac-окружение, включая
MLX и дополнительные инструменты сравнения PDF. Оно нужно только для
воспроизведения локального сервера и исторического сравнения экстракторов.

## 2. PDF, корпус и открытый набор одной командой

```bash
python tools/prepare_data.py --pdf /path/to/book.pdf --output local_data --dev-output . --check-reference
```

Скрипт последовательно запускает все четыре этапа без LLM и сети:

1. `prepare_pilot_corpus.py` извлекает первичный постраничный пилот.
2. `prepare_pilot_questions.py` преобразует сохранённые открытые примеры
   из Markdown в вопросы и page-only gold; экспортирует корневые dev-файлы.
3. `pdf_font_decoder.py` декодирует исходные глифы, шрифты и векторные черты.
   Профиль CM-Super T2A явно задан для кириллических шрифтов этого учебника.
4. `build_pdf_corpus.py` строит окончательные блоки, формулы, пространственные
   сетки и provenance. Первичный корпус используется только для сравнения.

Основные результаты:

| Путь | Содержание |
| --- | --- |
| `local_data/pilot/` | Пилотный корпус, страницы, вопросы и gold |
| `local_data/font_analysis/` | Глифы, шрифты, черты и отчёт декодирования |
| `local_data/corpus_v2/corpus.jsonl` | 1066 исходных блоков на 147 страницах |
| `local_data/corpus_v2/` | Происхождение блоков, геометрия и диагностика |
| `dev_questions.jsonl` | 15 вопросов в формате id/question/options |
| `dev_gold.jsonl` | Ответы, категории, обоснования и эталонные страницы |
| `local_data/preparation_manifest.json` | Хеши, время и результат сверки |

`--check-reference` сверяет хеши корпуса, вопросов и gold с опубликованным
`submission_manifest.json`; при расхождении команда завершается с ошибкой.
Для другого исходного набора этот флаг можно опустить. Подготовка создаёт
несколько сотен мегабайт локальной геометрии; эти файлы исключены из Git.

28 октября 2026 эта команда выполнена из другой рабочей папки в отдельном
окружении Python 3.11.15 с десятью пакетами из `requirements-pdf.txt`.
За 27,237 секунды восстановлены 147 страниц и 1066 блоков; контрольные
суммы корпуса, dev-вопросов и gold совпали с исходным прогоном. Модели
не скачивались, LLM-вызовов не было. Временные данные и окружение удалены.

Те же этапы можно выполнить отдельно:

```bash
python tools/prepare_pilot_corpus.py --pdf /path/to/book.pdf --output local_data/pilot
python tools/prepare_pilot_questions.py --source materials/tfkp_15_qa.md --output local_data/pilot --dev-output .
python tools/pdf_font_decoder.py --pdf /path/to/book.pdf --type3-encoding cm-super-t2a --output-dir local_data/font_analysis
python tools/build_pdf_corpus.py --pdf /path/to/book.pdf --glyphs local_data/font_analysis/glyph_pages.jsonl --old-corpus local_data/pilot/pilot_corpus.jsonl --output local_data/corpus_v2
```

Dev-файлы относятся к открытому локальному пилоту. Gold помечен как
`page_only_pilot_no_official_citation_f1`: по нему оцениваются варианты,
отказы и покрытие страниц, а не официальный F1 ссылок. Восемь авторских
вопросов и их эталоны — самостоятельные исходные файлы `own_questions.jsonl`
и `own_gold.jsonl`; они не генерируются при запуске RAG.

## 3. Эмбеддер и поисковый индекс

```bash
python tools/download_embedder.py --config config.json
python build_index.py --corpus local_data/corpus_v2/corpus.jsonl --output local_data/index_hybrid --config config.json
```

Загружается закреплённая ревизия EmbeddingGemma q8. При отсутствии локальной
Qwen сборщик скачает только её токенизатор фиксированной ревизии; веса
Qwen для индексации не нужны. Все файлы LanceDB и энкодера помещаются внутрь
папки индекса. Сборка не вызывает генеративные модели.

## 4. Итоговые ответы через предоставленный endpoint

```bash
export LLM_BASE_URL=http://127.0.0.1:8080/v1
export LLM_MODEL=/absolute/path/to/models/Qwen3.5-4B
export LLM_API_KEY=local
export SUBMISSION_ID=rag-hw1-reproduction

python predict.py --index local_data/index_hybrid --questions dev_questions.jsonl --output local_data/reproduced_run/answers.jsonl --trace local_data/reproduced_run/trace.jsonl --config config.json
python evaluate_pilot.py --questions dev_questions.jsonl --gold dev_gold.jsonl --answers local_data/reproduced_run/answers.jsonl --trace local_data/reproduced_run/trace.jsonl --index local_data/index_hybrid --output local_data/reproduced_run/metrics.json
```

Для внешнего сервера подставьте выданные URL, model ID и ключ. Используется
Chat Completions API с заголовками X-Question-ID и X-Submission-ID. При
ответах сеть нужна только для endpoint. `runtime.json` код не читает;
все параметры находятся в `config.json`. Если курс задаёт иные значения,
сверьте конфигурацию перед новым прогоном.

Две стадии генерации, все фактические usage и исходные ответы сохраняются
в локальном trace. Парсер восстанавливает только однозначные названия полей,
не меняет выбранные варианты по gold. Никаких автоматических повторов нет.

## 5. Baseline и две абляции

Первый baseline использует другой профиль и отдельный индекс:

```bash
python build_index.py --corpus local_data/corpus_v2/corpus.jsonl --output local_data/index_baseline --config config.baseline.json
export SUBMISSION_ID=rag-hw1-baseline-reproduction
python predict.py --index local_data/index_baseline --questions dev_questions.jsonl --output local_data/baseline_run/answers.jsonl --trace local_data/baseline_run/trace.jsonl --config config.baseline.json
python evaluate_pilot.py --questions dev_questions.jsonl --gold dev_gold.jsonl --answers local_data/baseline_run/answers.jsonl --trace local_data/baseline_run/trace.jsonl --index local_data/index_baseline --output local_data/baseline_run/metrics.json
```

Без контекста, без dense и с одним вызовом:

```bash
python tools/run_comparison.py --output local_data/comparison_reproduced --index local_data/index_hybrid --questions dev_questions.jsonl --gold dev_gold.jsonl
```

Три запуска идут последовательно. Скрипт не перезаписывает существующую
папку. Gold передаётся только оценщику после предсказания. Результаты пяти
сохранённых конфигураций находятся в `experiments.csv` и
[отчёте сравнения](comparison_report.md). Большие логи абляций в Git не нужны.

## 6. Экспорт итогового прогона для сдачи

```bash
python tools/export_submission.py --run local_data/reproduced_run --questions dev_questions.jsonl --corpus local_data/corpus_v2/corpus.jsonl
```

Команда копирует фактические ответы в `dev_answers.jsonl`, создаёт компактный
`dev_trace.jsonl` со всеми расходами и хешами запросов, обновляет manifest.
Тексты фрагментов и большие исходные ответы остаются в локальном trace.
Экспорт проверяет совпадение вопросов, корпуса и конфигурации с прогоном.
Опубликованные исторические ответы после ручного аудита не изменялись.

## 7. Необязательный локальный сервер на Apple Silicon

При предоставленном endpoint этот раздел можно пропустить. Для исходного
локального запуска на Mac:

```bash
python -m pip install -r requirements-local.txt
python tools/download_qwen.py --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
python tools/serve_qwen.py
```

В другом терминале используйте endpoint `http://127.0.0.1:8080/v1` и model ID
с абсолютным путём к `models/Qwen3.5-4B`. Эмбеддинги работают на CPU,
Qwen — на Apple GPU через MLX. Загрузка Qwen занимает несколько гигабайт.

Корпус, геометрия, индексы, LanceDB, веса, окружение и кэши остаются
локальными. После очистки 23 октября веса, окружение и индексы удалены;
сохранённые команды позволяют создать их заново.

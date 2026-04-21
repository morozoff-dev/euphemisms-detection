# Detection of Synthetic Euphemistic Drug Mentions in Russian Texts

## Что это за проект

Это исследовательский прототип для детекции синтетически подставленных эвфемистических обозначений наркотических веществ в русскоязычных предложениях с помощью sequence labeling.

Важно: проект не решает задачу real-world slang detection. Корпус и постановка задачи рассматриваются как controlled synthetic benchmark.

## Идея пайплайна

Проект устроен как трёхшаговый pipeline:

1. Из сырых текстов наркотической тематики строится synthetic positive dataset:
   - находятся канонические названия веществ;
   - они заменяются на эвфемизмы с сохранением морфологических свойств;
   - сохраняются char-level спаны замен.
2. Из synthetic positives и отрицательных примеров собирается trainable BIO dataset:
   - positive samples берутся из `outputs/synthetic/data.json`;
   - negative samples берутся из `data/negatives.txt`;
   - текст разбивается на word-level токены;
   - по char-level спанам строятся BIO-теги;
   - данные режутся на `train/val/test`;
   - сохраняются готовые `jsonl`-файлы для обучения.
3. На готовом BIO dataset обучается baseline token-classification модель:
   - берётся `ModernBERT` / `RuModernBERT`;
   - word-level BIO-теги выравниваются на subword-токены;
   - считается token-level и span-level F1;
   - сохраняются лучший чекпоинт, метрики, предсказания на `val/test` и отдельный человеко-читаемый `val`-лог FP/FN.

## Структура проекта

- `data/` — входные данные
- `src/data/` — загрузка и очистка текста
- `src/synthetic/` — генерация synthetic positive dataset
- `src/models/` — загрузка transformer-моделей и токенизаторов
- `src/training/` — подготовка BIO dataset и обучение baseline модели
- `outputs/synthetic/` — сгенерированный synthetic JSON
- `outputs/training/` — подготовленные splits, чекпоинты, метрики и предсказания

## Форматы данных

### Synthetic positive dataset

После запуска `src.synthetic` создаётся файл `outputs/synthetic/data.json`.

Каждый sample содержит:

- `text` — исходный текст;
- `replaced_text` — текст после замены названия вещества на эвфемизм;
- `euphemisms` — список аннотаций со span-ами, например:
  - `start`, `end` — границы эвфемизма в `replaced_text`;
  - `target_word`, `target_lemma`;
  - `base_euphemism`, `euphemism`.

### Training BIO dataset

После запуска `src.training` создаются:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `manifest.json`

Каждая строка в `*.jsonl` содержит:

- `sample_id`
- `source` — `positive` или `negative`
- `text`
- `tokens` — word-level токены
- `bio_tags` — BIO-разметка для токенов
- `entities` — entity spans в char-level формате

`manifest.json` хранит служебную информацию о сборке датасета:

- какой `seed` использовался;
- какие входные файлы брались;
- сколько примеров было до и после sampling;
- как распределились данные по `train/val/test`;
- сколько было alignment warning'ов и отброшенных sample'ов.

## Установка

В репозитории уже предполагается локальное окружение `venv/`.

Если окружение уже создано, можно просто использовать:

```bash
venv/bin/python
```

Базовые зависимости перечислены в `requirements.txt`.

Для обучения baseline модели нужен PyTorch. Если вы обучаете на GPU, лучше сначала поставить подходящую сборку `torch` по инструкции PyTorch для вашей CUDA-версии, а затем доустановить зависимости проекта:

```bash
venv/bin/pip install -r requirements.txt
```

Для `ModernBERT` нужна версия `transformers>=4.48.0`.

## Как запустить

### 1. Сгенерировать synthetic positive dataset

Из корня проекта:

```bash
venv/bin/python -m src.synthetic
```

По умолчанию будут использованы:

- `data/drug_texts_small.txt`
- `data/real_euphemisms.txt`
- `data/target_keywords_forms_drug.txt`

Результат будет записан в:

```bash
outputs/synthetic/data.json
```

### 2. Подготовить BIO dataset для обучения

Минимальный запуск:

```bash
venv/bin/python -m src.training
```

По умолчанию будут использованы:

- positives: `outputs/synthetic/data.json`
- negatives: `data/negatives.txt`
- output dir: `outputs/training/bio_dataset`
- split ratios: `0.8 / 0.1 / 0.1`
- `seed=42`

Пример запуска на части данных:

```bash
venv/bin/python -m src.training \
  --positive-limit 20000 \
  --negative-limit 20000 \
  --output-dir outputs/training/run_01 \
  --seed 42
```

Поддерживаются параметры:

- `--positive-limit`
- `--negative-limit`
- `--positive-fraction`
- `--negative-fraction`
- `--train-ratio`
- `--val-ratio`
- `--test-ratio`
- `--seed`

### 3. Обучить baseline ModernBERT / RuModernBERT

Основной запуск на русском baseline:

```bash
venv/bin/python -m src.training.train \
  --model-name deepvk/RuModernBERT-base \
  --output-dir outputs/training/rumodernbert_base_run01 \
  --epochs 3 \
  --train-batch-size 8 \
  --eval-batch-size 16 \
  --max-length 256
```

CLI по умолчанию использует `patched-tokenizer` для `deepvk/RuModernBERT-*`, потому что это важнее для NER/sequence labeling, чем стандартный tokenizer revision.

Быстрый smoke test на небольшом подмножестве:

```bash
venv/bin/python -m src.training.train \
  --model-name deepvk/RuModernBERT-small \
  --output-dir outputs/training/rumodernbert_smoke \
  --epochs 1 \
  --train-batch-size 4 \
  --eval-batch-size 8 \
  --max-length 256 \
  --max-train-samples 512 \
  --max-val-samples 128 \
  --max-test-samples 128
```

Если автоматическое скачивание модели недоступно, можно вручную скачать:

- model checkpoint: `deepvk/RuModernBERT-base`
- tokenizer: `deepvk/RuModernBERT-base`, revision `patched-tokenizer`

После этого CLI можно запускать на локальных папках:

```bash
venv/bin/python -m src.training.train \
  --model-name /path/to/local/model \
  --tokenizer-name /path/to/local/tokenizer \
  --output-dir outputs/training/rumodernbert_local_run
```

Основные параметры training CLI:

- `--model-name`
- `--tokenizer-name`
- `--model-revision`
- `--tokenizer-revision`
- `--cache-dir`
- `--max-length`
- `--overflow-handling`
- `--epochs`
- `--train-batch-size`
- `--eval-batch-size`
- `--learning-rate`
- `--weight-decay`
- `--warmup-ratio`
- `--grad-accumulation-steps`
- `--max-grad-norm`
- `--device`
- `--mixed-precision`
- `--max-train-samples`
- `--max-val-samples`
- `--max-test-samples`

После запуска `src.training.train` сохраняются:

- `run_config.json`
- `best_model/`
- `best_model_metrics.json`
- `metrics.json`
- `training_summary.json`
- `predictions/val.jsonl`
- `predictions/test.jsonl`
- `analysis/val_fp_fn.md`

В `metrics.json` сохраняются:

- token-level precision / recall / F1 / accuracy;
- span-level precision / recall / F1;
- `val` и `test` метрики лучшего чекпоинта.

В `analysis/val_fp_fn.md` сохраняется читаемый markdown-отчёт по `val`:

- в отчёт попадают только sample-ы с ошибками `FP` или `FN`;
- для каждого sample показываются исходный `Text`, строки `Gold` и `Pred`;
- entity span-ы подсвечиваются как `[[...]]`;
- отдельно списком выписываются только `FP` и `FN` с token offsets.

## Что уже реализовано

- загрузка и очистка raw texts;
- поиск канонических форм по словарю;
- морфологически согласованная подстановка эвфемизмов;
- генерация synthetic positive dataset;
- подготовка BIO training dataset;
- baseline training loop для `ModernBERT` / `RuModernBERT`;
- token-level и span-level evaluation;
- сохранение лучшего чекпоинта, predictions и metrics;
- отдельный читаемый `val`-лог FP/FN для span-level error analysis;
- reproducible sampling и split по `seed`;
- предупреждения в терминал, если char-level span нечётко совпадает с границами токенов;
- `run_config.json` для воспроизводимого baseline training run.

## Что пока не реализовано

- experimental context-only head;
- error analysis.

## Программное использование

Если нужно загрузить уже подготовленный split в Python-коде:

```python
from src.training import PreparedBioDataset

train_ds = PreparedBioDataset.from_directory(
    "outputs/training/bio_dataset",
    split="train",
)
```

## Замечание по документации

При добавлении новых модулей, команд или этапов пайплайна нужно обновлять:

- `AGENTS.md` — как рабочий статус и агентные инструкции;
- `README.md` — как описание проекта для человека и инструкция по запуску.

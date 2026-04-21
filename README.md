# Detection of Synthetic Euphemistic Drug Mentions in Russian Texts

## Что это за проект

Это исследовательский прототип для детекции синтетически подставленных эвфемистических обозначений наркотических веществ в русскоязычных предложениях с помощью sequence labeling.

Важно: проект не решает задачу real-world slang detection. Корпус и постановка задачи рассматриваются как controlled synthetic benchmark.

## Идея пайплайна

Проект устроен как двухшаговый pipeline:

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

## Структура проекта

- `data/` — входные данные
- `src/data/` — загрузка и очистка текста
- `src/synthetic/` — генерация synthetic positive dataset
- `src/training/` — подготовка BIO train/val/test dataset
- `outputs/synthetic/` — сгенерированный synthetic JSON
- `outputs/training/` — подготовленные train/val/test splits

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

Зависимости перечислены в `requirements.txt`.

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

## Что уже реализовано

- загрузка и очистка raw texts;
- поиск канонических форм по словарю;
- морфологически согласованная подстановка эвфемизмов;
- генерация synthetic positive dataset;
- подготовка BIO training dataset;
- reproducible sampling и split по `seed`;
- предупреждения в терминал, если char-level span нечётко совпадает с границами токенов.

## Что пока не реализовано

- baseline model на ModernBERT / RuModernBERT;
- training loop;
- evaluation pipeline;
- token/span F1;
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

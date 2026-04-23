# Detection of Synthetic Euphemistic Drug Mentions in Russian Texts

## Что это за проект

Это исследовательский прототип для детекции синтетически подставленных эвфемистических обозначений наркотических веществ в русскоязычных предложениях с помощью sequence labeling.

Важно: проект не решает задачу real-world slang detection. Корпус и постановка задачи рассматриваются как controlled synthetic benchmark.

## Идея пайплайна

Проект устроен как трёхшаговый pipeline:

1. В `src.data_prep` из сырых positive и negative текстов строятся уже готовые split-файлы для датасета:
   - positive texts берутся из `data/drug_texts_small.txt`;
   - negative texts берутся из `data/negatives.txt`;
   - positive texts с упоминаниями target keywords сэмплируются, как и negatives;
   - positives и negatives независимо режутся на `train/val/test` с сохранением заданного соотношения;
   - target keywords заменяются частично: по умолчанию заменяется 50% найденных mentions, остальные остаются в тексте и тоже размечаются как сущности;
   - в `train/val` positives используют `generated_slang_euphemisms.txt` как replacement pool;
   - в `test` positives используют `generated_euphemisms.txt` как replacement pool;
   - сохраняются `train.json`, `val.json`, `test.json` и `manifest.json`.
2. В `src.bio` готовые split JSON-файлы переводятся в BIO-датасет:
   - входные `train.json`, `val.json`, `test.json` уже не пересэмпливаются и не режутся заново;
   - текст разбивается на word-level токены;
   - по char-level entity span'ам строятся BIO-теги;
   - сохраняются `train.jsonl`, `val.jsonl`, `test.jsonl` и `manifest.json`.
3. В `src.models.train` на готовом BIO-датасете обучается baseline token-classification модель:
   - берётся `ModernBERT` / `RuModernBERT`;
   - word-level BIO-теги выравниваются на subword-токены;
   - считается token-level и span-level F1;
   - для `test` дополнительно считаются отдельные masked-метрики по двум группам gold-сущностей:
     - `replacement_pool_only` — только synthetic replacements, пришедшие из test replacement pool;
     - `other_gold_entities_only` — остальные gold-сущности, то есть прежде всего незаменённые `target_keyword`;
   - после каждой эпохи в логах считаются метрики и на `val`, и на `test`;
   - сохраняются лучший чекпоинт, метрики, предсказания на `val/test` и отдельный человеко-читаемый `test`-лог FP/FN;
   - для каждого запуска автоматически создаётся отдельная run-папка в `outputs/models/`.

## Структура проекта

- `data/` — входные данные
- `src/data/` — загрузка и очистка текста
- `src/data_prep/` — подготовка data split'ов
- `src/bio/` — BIO-конвертация split JSON
- `src/models/` — загрузка transformer-моделей, метрики и baseline training
- `outputs/data_prep/` — подготовленные split JSON-файлы
- `outputs/bio/` — BIO-датасет
- `outputs/models/` — чекпоинты, метрики и предсказания моделей

## Форматы данных

### Prepared Split Dataset

После запуска `src.data_prep` создаются:

- `outputs/data_prep/splits/train.json`
- `outputs/data_prep/splits/val.json`
- `outputs/data_prep/splits/test.json`
- `outputs/data_prep/splits/manifest.json`

Каждый sample в `train.json` / `val.json` / `test.json` содержит:

- `sample_id`
- `source` — `positive` или `negative`
- `source_index`
- `text` — итоговый текст, который потом пойдёт в BIO-конвертацию
- `entities` — список entity span-ов

Для positive samples дополнительно сохраняются:

- `original_text`
- `euphemisms` — список положительных entity-аннотаций, включая:
  - synthetic replacements;
  - незаменённые target keyword mentions;
  - у каждой аннотации есть `annotation_kind` и `is_replaced`, чтобы было видно, была ли это подстановка или сущность пришла из исходного текста

`manifest.json` для data preparation stage хранит:

- какие входные файлы брались;
- какие euphemism vocab files использовались для `train/val/test`;
- какая доля target keyword mentions заменялась;
- сколько positive/negative примеров было до и после sampling;
- как распределились данные по `train/val/test`.

### BIO Dataset

После запуска `src.bio` создаются:

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
- `token_annotation_kinds` — список той же длины, что и `tokens`; для entity-токенов хранит `annotation_kind` (`synthetic_replacement`, `unchanged_target_keyword`), а для legacy split-файлов без явного `annotation_kind` может использовать fallback `other_gold_entity`; для остальных токенов хранит `null`

`manifest.json` для BIO-конвертации хранит:

- какие входные split JSON-файлы брались;
- сколько sample-ов было во входных и выходных split'ах;
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
Для TensorBoard-графиков `requirements.txt` теперь также включает `tensorboard`.

## Как запустить

### 1. Подготовить Dataset Splits

Из корня проекта:

```bash
venv/bin/python -m src.data_prep
```

По умолчанию будут использованы:

- `data/drug_texts_small.txt`
- `data/negatives.txt`
- `data/target_keywords_forms_drug.txt`
- train/val positive euphemisms:
  - `data/generated_slang_euphemisms.txt`
- test positive euphemisms:
  - `data/generated_euphemisms.txt`
- `target_replacement_fraction=0.5`
- `positive_limit=10000`
- `negative_limit=2000`

Результат будет записан в:

```bash
outputs/data_prep/splits/
```

Новый default pipeline делает всё на этапе подготовки данных:

- сэмплирует positive и negative source texts;
- делит их на `train/val/test`;
- смешивает positives и negatives внутри каждого split;
- заменяет только часть target keyword mentions, а остальные target mentions тоже размечает как сущности;
- в `train/val` использует `generated_slang_euphemisms.txt` как replacement pool;
- в `test` использует `generated_euphemisms.txt` как replacement pool.

Явный запуск этого сценария:

```bash
venv/bin/python -m src.data_prep \
  --positive-limit 10000 \
  --negative-limit 2000 \
  --train-euphemisms-paths data/generated_slang_euphemisms.txt \
  --test-euphemisms-paths data/generated_euphemisms.txt \
  --output-dir outputs/data_prep/splits \
  --target-replacement-fraction 0.5 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

Если нужно поменять negatives или euphemism vocab files, это теперь делается здесь, на этапе подготовки данных.

### 2. Преобразовать Split JSON в BIO

Минимальный запуск:

```bash
venv/bin/python -m src.bio
```

По умолчанию будут использованы:

- input dir: `outputs/data_prep/splits`
- output dir: `outputs/bio`

`src.bio` не делает sampling и split. Он только переводит уже готовые `train.json`, `val.json`, `test.json` из этапа подготовки данных в BIO `jsonl`.

Если у вас уже были старые `outputs/bio/*.jsonl`, после обновления кода их нужно один раз пересобрать через `src.bio`, чтобы в датасет попали `token_annotation_kinds` для новых test subset metrics.

Обычный запуск:

```bash
venv/bin/python -m src.bio \
  --input-dir outputs/data_prep/splits \
  --output-dir outputs/bio
```

Поддерживаются параметры:

- `--input-dir`
- `--train-path`
- `--val-path`
- `--test-path`
- `--output-dir`

### 3. Обучить baseline ModernBERT / RuModernBERT

Основной запуск на русском baseline:

```bash
venv/bin/python -m src.models.train \
  --model-name deepvk/RuModernBERT-base \
  --epochs 3 \
  --train-batch-size 8 \
  --eval-batch-size 16 \
  --max-length 256
```

При каждом запуске `src.models.train` автоматически создаётся новая папка вида `outputs/models/rumodernbert_base_04_22_15_37`, где суффикс — это `месяц_день_час_минута` времени старта train.

Параметр `--output-dir` теперь задаёт базовую директорию для таких auto-generated run-папок.

CLI по умолчанию использует `patched-tokenizer` для `deepvk/RuModernBERT-*`, потому что это важнее для NER/sequence labeling, чем стандартный tokenizer revision.

TensorBoard-логи теперь пишутся автоматически для каждого training run в:

```bash
outputs/models/<run_name>/tensorboard/
```

Запустить просмотр можно так:

```bash
venv/bin/tensorboard --logdir outputs/models
```

В TensorBoard сохраняются:

- `epoch/train/*`, `epoch/val/*`, `epoch/test/*` — epoch-level метрики для train/val/test;
- `epoch/test/subsets/*` — subset-метрики для `replacement_pool_only` и `other_gold_entities_only`;
- в TensorBoard Custom Scalars дополнительно собираются сравнительные графики:
  - `train/val/test loss`;
  - `val/test token F1`;
  - `val/test span F1`;
  - `test subset span F1`;
- `final/val/*` и `final/test/*` — финальные метрики лучшего чекпоинта.

Быстрый smoke test на небольшом подмножестве:

```bash
venv/bin/python -m src.models.train \
  --model-name deepvk/RuModernBERT-small \
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
venv/bin/python -m src.models.train \
  --model-name /path/to/local/model \
  --tokenizer-name /path/to/local/tokenizer \
  --output-dir outputs/models_local
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

После запуска `src.models.train` в автоматически созданную папку `outputs/models/<run_name>/` сохраняются:

- `run_config.json`
- `tensorboard/`
- `best_model/`
- `best_model_metrics.json`
- `metrics.json`
- `training_summary.json`
- `predictions/val.jsonl`
- `predictions/test.jsonl`
- `analysis/test_fp_fn.md`

В `metrics.json` сохраняются:

- token-level precision / recall / F1 / accuracy;
- span-level precision / recall / F1;
- `val` и `test` метрики лучшего чекпоинта;
- для `test` также сохраняется `subsets`:
  - `replacement_pool_only` — метрики только по gold-спанам типа `synthetic_replacement`, то есть по сущностям, пришедшим из того vocabulary pool, который был передан в `--test-euphemisms-paths`;
  - `other_gold_entities_only` — метрики по остальным gold-сущностям в `test` (`unchanged_target_keyword`, а для legacy split-файлов также fallback `other_gold_entity`).

`training_summary.json`, `metrics.json` и остальные JSON-артефакты продолжают сохраняться как раньше; TensorBoard-директория добавляется дополнительно и не заменяет существующие метрики.

В `analysis/test_fp_fn.md` сохраняется читаемый markdown-отчёт по `test`:

- в отчёт попадают только sample-ы с ошибками `FP` или `FN`;
- для каждого sample показываются исходный `Text`, строки `Gold` и `Pred`;
- entity span-ы подсвечиваются как `[[...]]`;
- отдельно списком выписываются только `FP` и `FN` с token offsets.

## Что уже реализовано

- загрузка и очистка raw texts;
- поиск канонических форм по словарю;
- морфологически согласованная подстановка эвфемизмов;
- подготовка data split dataset с positives и negatives;
- преобразование split JSON в BIO-датасет;
- baseline training loop для `ModernBERT` / `RuModernBERT` в `src.models`;
- token-level и span-level evaluation;
- сохранение лучшего чекпоинта, predictions и metrics;
- отдельный читаемый `val`-лог FP/FN для span-level error analysis;
- reproducible sampling и split по `seed` на этапе подготовки данных;
- предупреждения в терминал, если char-level span нечётко совпадает с границами токенов;
- `run_config.json` для воспроизводимого baseline training run.

## Что пока не реализовано

- experimental context-only head;
- error analysis.

## Программное использование

Если нужно загрузить уже подготовленный split в Python-коде:

```python
from src.bio import BioDataset

train_ds = BioDataset.from_directory(
    "outputs/bio",
    split="train",
)
```

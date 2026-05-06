# Detection of Synthetic Euphemistic Drug Mentions in Russian Texts

## Что это за проект

Это исследовательский прототип для детекции синтетически подставленных эвфемистических обозначений наркотических веществ в русскоязычных предложениях с помощью sequence labeling.

Важно: проект не решает задачу real-world slang detection. Корпус и постановка задачи рассматриваются как controlled synthetic benchmark.

## Идея пайплайна

Проект устроен как трёхшаговый pipeline:

1. В `src.data_prep` из сырых positive и negative текстов строятся уже готовые split-файлы для датасета:
   - positive texts берутся из `data/drug_texts_small.txt`;
   - negative texts берутся из `data/negatives.txt`;
   - отдельные extra negatives с эвфемизмами по умолчанию берутся из `data/train_val_negatives_with_euphemisms.txt` для `train/val` и `data/test_negatives_with_euphemisms.txt` для `test`;
   - перед language filter тексты проходят annotation-oriented preprocessing: `NFKC`, удаление zero-width/HTML/entity/emoji, замена `URL` / `email` / `@mention` / телефонов / длинных чисел на маркеры;
   - на раннем preprocessing этапе сохраняются только тексты, для которых `cld2` определяет основной язык как русский (`ru`);
   - если в тексте более 50% букв находятся в верхнем регистре, такой текст полностью переводится в нижний регистр;
   - пунктуация сохраняется, а числовые токены длиной до 3 цифр остаются в тексте как есть;
   - positive texts с упоминаниями target keywords сэмплируются, как и negatives;
   - positives и negatives независимо режутся на `train/val/test` с сохранением заданного соотношения;
   - extra `train_val` negatives режутся только между `train` и `val`, extra `test` negatives целиком добавляются только в `test`;
   - target keywords заменяются частично: по умолчанию заменяется 50% найденных mentions, остальные остаются в тексте и тоже размечаются как сущности;
   - в `train/val` positives используют `train_val_euphemisms.txt` как replacement pool;
   - в `test` positives используют `test_euphemisms.txt` как replacement pool;
   - сохраняются `train.json`, `val.json`, `test.json` и `manifest.json`.
2. В `src.bio` готовые split JSON-файлы переводятся в бинарный token-label датасет:
   - входные `train.json`, `val.json`, `test.json` уже не пересэмпливаются и не режутся заново;
   - текст разбивается на токены, которые сохраняют слова, маркеры, пунктуацию и числа до 3 цифр;
   - по char-level entity span'ам строятся два token-level тега: `O` и `EUPHEMISM`;
   - сохраняются `train.jsonl`, `val.jsonl`, `test.jsonl` и `manifest.json`.
3. В `src.models.train` на готовом token-label датасете обучается бинарная token-classification модель:
   - берётся `ModernBERT` / `RuModernBERT`;
   - token-level теги из `tokens` выравниваются на subword-токены;
   - доступны три режима головы: `baseline`, `neighbor`, `combined`;
   - все три режима обучаются как binary classifier с одним positive-logit и `BCEWithLogitsLoss`, но наружу возвращают logits формы `[batch, seq, 2]`, чтобы существующие `argmax`-метрики работали без отдельной ветки;
   - считается token-level и span-level F1;
   - для `test` дополнительно считаются отдельные masked-метрики по двум группам gold-сущностей:
     - `replacement_pool_only` — только synthetic replacements, пришедшие из test replacement pool;
     - `other_gold_entities_only` — остальные gold-сущности, то есть прежде всего незаменённые `target_keyword`;
   - для extra negative groups отдельно считаются FP-oriented counts без изменения общих token/span метрик;
   - после каждой эпохи в логах считаются метрики и на `val`, и на `test`;
   - сохраняются лучший чекпоинт, метрики, предсказания на `val/test` и отдельный человеко-читаемый `test`-лог FP/FN;
   - для каждого запуска автоматически создаётся отдельная run-папка в `outputs/models/`.

## Структура проекта

- `data/` — входные данные
- `src/data/` — загрузка и очистка текста
- `src/data_prep/` — подготовка data split'ов
- `src/bio/` — token-label конвертация split JSON
- `src/models/` — загрузка transformer-моделей, custom heads, метрики и training
- `src/evaluation/` — evaluation CLI для verified real-euphemism JSON
- `outputs/data_prep/` — подготовленные split JSON-файлы
- `outputs/bio/` — token-label датасет
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
- `text` — итоговый текст, который потом пойдёт в token-label конвертацию
- `entities` — список entity span-ов
- `negative_group` — есть только у extra negative samples, например `negative_euphemism_match`

Для positive samples дополнительно сохраняются:

- `original_text`
- `euphemisms` — список положительных entity-аннотаций, включая:
  - synthetic replacements;
  - незаменённые target keyword mentions;
  - у каждой аннотации есть `annotation_kind` и `is_replaced`, чтобы было видно, была ли это подстановка или сущность пришла из исходного текста

`manifest.json` для data preparation stage хранит:

- какие входные файлы брались;
- какие правила раннего preprocessing применялись (`cld2` language filter, lowercasing mostly-uppercase текстов и marker-based text normalization);
- какие euphemism vocab files использовались для `train/val/test`;
- сколько positive/negative текстов осталось после preprocessing;
- сколько extra negative текстов осталось после preprocessing;
- какая доля target keyword mentions заменялась;
- сколько positive/negative примеров было до и после sampling;
- как распределились данные по `train/val/test`.

### Token-label Dataset

После запуска `src.bio` создаются:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `manifest.json`

Каждая строка в `*.jsonl` содержит:

- `sample_id`
- `source` — `positive` или `negative`
- `text`
- `tokens` — предтокенизированные единицы для NER: слова, маркеры, пунктуация и числа до 3 цифр
- `bio_tags` — token-level разметка для токенов; поле сохранило старое имя для совместимости, но теперь содержит только `O` или `EUPHEMISM`
- `entities` — entity spans в char-level формате
- `negative_group` — optional group marker для extra negative samples
- `token_annotation_kinds` — список той же длины, что и `tokens`; для entity-токенов хранит `annotation_kind` (`synthetic_replacement`, `unchanged_target_keyword`), а для legacy split-файлов без явного `annotation_kind` может использовать fallback `other_gold_entity`; для остальных токенов хранит `null`

`manifest.json` для token-label конвертации хранит:

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

Для обучения модели нужен PyTorch. Если вы обучаете на GPU, лучше сначала поставить подходящую сборку `torch` по инструкции PyTorch для вашей CUDA-версии, а затем доустановить зависимости проекта:

```bash
venv/bin/pip install -r requirements.txt
```

Для `ModernBERT` нужна версия `transformers>=4.48.0`.
Для TensorBoard-графиков `requirements.txt` теперь также включает `tensorboard`.
Для ранней фильтрации русскоязычных текстов на этапе `src.data_prep` `requirements.txt` теперь также включает `pycld2`.

## Как запустить

### 1. Подготовить Dataset Splits

Из корня проекта:

```bash
venv/bin/python -m src.data_prep
```

По умолчанию будут использованы:

- `data/drug_texts_small.txt`
- `data/negatives.txt`
- extra train/val negatives:
  - `data/train_val_negatives_with_euphemisms.txt`
- extra test negatives:
  - `data/test_negatives_with_euphemisms.txt`
- `data/target_keywords_forms_drug.txt`
- train/val positive euphemisms:
  - `data/train_val_euphemisms.txt`
- test positive euphemisms:
  - `data/test_euphemisms.txt`
- `target_replacement_fraction=0.5`
- `positive_limit=10000`
- `negative_limit=2000`
- `extra_negative_group_name=negative_euphemism_match`

Результат будет записан в:

```bash
outputs/data_prep/splits/
```

Новый default pipeline делает всё на этапе подготовки данных:

- прогоняет source texts через annotation-oriented preprocessing с маркерами `<URL>`, `<EMAIL>`, `<USER>`, `<PHONE>`, `<BIGNUM>`;
- оставляет только те positive и negative source texts, для которых `cld2` определяет основной язык как русский;
- если в тексте больше 50% букв в верхнем регистре, полностью переводит его в lower-case;
- сохраняет пунктуацию и короткие числовые токены, чтобы они доходили до token-label датасета и BERT;
- сэмплирует positive и negative source texts;
- делит их на `train/val/test`;
- добавляет extra negatives поверх обычных negatives: `train_val` файл делится только между `train` и `val`, `test` файл целиком попадает только в `test`;
- помечает эти rows как `source="negative"` и `negative_group="negative_euphemism_match"`;
- смешивает positives и negatives внутри каждого split;
- заменяет только часть target keyword mentions, а остальные target mentions тоже размечает как сущности;
- в `train/val` использует `train_val_euphemisms.txt` как replacement pool;
- в `test` использует `test_euphemisms.txt` как replacement pool.

Явный запуск этого сценария:

```bash
venv/bin/python -m src.data_prep \
  --positive-limit 10000 \
  --negative-limit 2000 \
  --extra-negative-train-val-path data/train_val_negatives_with_euphemisms.txt \
  --extra-negative-test-path data/test_negatives_with_euphemisms.txt \
  --extra-negative-group-name negative_euphemism_match \
  --train-euphemisms-paths data/train_val_euphemisms.txt \
  --test-euphemisms-paths data/test_euphemisms.txt \
  --output-dir outputs/data_prep/splits \
  --target-replacement-fraction 0.5 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

Если нужно поменять negatives или euphemism vocab files, это теперь делается здесь, на этапе подготовки данных.
Обычная логика `--negatives-path` / `--negative-limit` не меняется; extra negative group добавляется отдельно. Для полного legacy-запуска без extra group используйте `--disable-extra-negative-group`.

### 2. Преобразовать Split JSON в token-label датасет

Минимальный запуск:

```bash
venv/bin/python -m src.bio
```

По умолчанию будут использованы:

- input dir: `outputs/data_prep/splits`
- output dir: `outputs/bio`

`src.bio` не делает sampling и split. Он только переводит уже готовые `train.json`, `val.json`, `test.json` из этапа подготовки данных в token-label `jsonl`, сохраняя в `tokens` слова, маркеры, пунктуацию и числа до 3 цифр.

Если у вас уже были старые `outputs/bio/*.jsonl`, после обновления кода их нужно один раз пересобрать через `src.bio`, чтобы теги стали бинарными (`O` / `EUPHEMISM`) и в датасет попали `token_annotation_kinds` для test subset metrics.

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

### 3. Обучить ModernBERT / RuModernBERT

Основной запуск на русском baseline head:

```bash
venv/bin/python -m src.models.train \
  --model-name deepvk/RuModernBERT-base \
  --head-mode baseline \
  --epochs 3 \
  --train-batch-size 8 \
  --eval-batch-size 16 \
  --max-length 256
```

Доступны три режима:

- `baseline` — hidden state первого subword каждого word token -> dropout -> `Linear(hidden, 1)`;
- `neighbor` — для word-start токена берутся contextualized states соседних word-start токенов; два соседа усредняются, один сосед берётся как есть, для single-word input используется zero-vector;
- `combined` — `alpha * baseline_logit + (1 - alpha) * neighbor_logit`, где `alpha = sigmoid(raw_alpha)`, а стартовое значение `alpha` задаётся через `--initial-alpha` (`0.5` по умолчанию, что соответствует `raw_alpha = 0.0`).

Для `combined` параметр `raw_alpha` остаётся обучаемым, но получает отдельный learning rate через `--alpha-learning-rate` (`1e-3` по умолчанию), без weight decay.

Важно: `neighbor` не является строгим context-only head в смысле полностью независимого контекста. Он не использует hidden state текущего word-start токена напрямую, но соседние hidden states уже contextualized encoder states и могут содержать информацию о текущем токене через self-attention.

Запуски experimental heads отличаются только `--head-mode`:

```bash
venv/bin/python -m src.models.train \
  --model-name deepvk/RuModernBERT-base \
  --head-mode neighbor
```

```bash
venv/bin/python -m src.models.train \
  --model-name deepvk/RuModernBERT-base \
  --head-mode combined \
  --initial-alpha 0.5 \
  --alpha-learning-rate 1e-2
```

Локальный smoke test custom heads без скачивания внешних весов:

```bash
venv/bin/python scripts/smoke_custom_heads.py
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
- `epoch/model/alpha` — значение `alpha` для `combined` head по эпохам;
- `epoch/test/subsets/*` — subset-метрики для `replacement_pool_only` и `other_gold_entities_only`;
- в TensorBoard Custom Scalars дополнительно собираются сравнительные графики:
  - `train/val/test loss`;
  - `val/test token F1`;
  - `val/test span F1`;
  - `combined head alpha`;
  - `test subset span F1`;
- `final/val/*` и `final/test/*` — финальные метрики лучшего чекпоинта.

Быстрый smoke test на небольшом подмножестве:

```bash
venv/bin/python -m src.models.train \
  --model-name deepvk/RuModernBERT-small \
  --head-mode baseline \
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
  --head-mode baseline \
  --output-dir outputs/models_local
```

Основные параметры training CLI:

- `--model-name`
- `--tokenizer-name`
- `--head-mode`
- `--model-revision`
- `--tokenizer-revision`
- `--cache-dir`
- `--max-length`
- `--overflow-handling`
- `--epochs`
- `--train-batch-size`
- `--eval-batch-size`
- `--learning-rate`
- `--alpha-learning-rate`
- `--initial-alpha`
- `--weight-decay`
- `--warmup-ratio`
- `--grad-accumulation-steps`
- `--max-grad-norm`
- `--device`
- `--mixed-precision`
- `--max-train-samples`
- `--max-val-samples`
- `--max-test-samples`
- `--best-checkpoint-metric`
- `--best-checkpoint-tie-breaker-metric`

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
- `analysis/test_negative_euphemism_match_fp.md`, если в test есть эта extra negative group

Лучший чекпоинт теперь выбирается только по `test`:

- primary metric: `test_span_f1`
- tie-breaker: `test_token_f1`
- `val`-метрики больше не участвуют в выборе `best_model`

При необходимости можно менять только метрики выбора:

- `--best-checkpoint-metric span_f1|token_f1`
- `--best-checkpoint-tie-breaker-metric span_f1|token_f1`

В `metrics.json` сохраняются:

- token-level precision / recall / F1 / accuracy;
- span-level precision / recall / F1;
- `head_mode`, `model_architecture`, `positive_label_id` и `alpha` для `combined`;
- `best_checkpoint_selection` с политикой выбора `best_model` и score выбранного чекпоинта;
- `val` и `test` метрики лучшего чекпоинта;
- для `test` также сохраняется `subsets`:
  - `replacement_pool_only` — метрики только по gold-спанам типа `synthetic_replacement`, то есть по сущностям, пришедшим из того vocabulary pool, который был передан в `--test-euphemisms-paths`;
  - `other_gold_entities_only` — метрики по остальным gold-сущностям в `test` (`unchanged_target_keyword`, а для legacy split-файлов также fallback `other_gold_entity`).
- для `val` / `test` при наличии extra negative groups сохраняется `negative_groups.<group_name>`:
  - `samples`, `total_tokens`;
  - `token_fp`, `span_fp`, `predicted_spans`;
  - `samples_with_token_fp`, `samples_with_span_fp`.

`run_config.json`, `training_summary.json`, `metrics.json` и `best_model_metrics.json` теперь также содержат `head_mode`, `model_architecture`, `positive_label_id`, `alpha` для `combined` и `best_checkpoint_selection`; TensorBoard-директория добавляется дополнительно и не заменяет существующие метрики.

В `analysis/test_fp_fn.md` сохраняется читаемый markdown-отчёт по `test`:

- в отчёт попадают только sample-ы с ошибками `FP` или `FN`;
- для каждого sample показываются исходный `Text`, строки `Gold` и `Pred`;
- entity span-ы подсвечиваются как `[[...]]`;
- отдельно списком выписываются только `FP` и `FN` с token offsets.

Для extra negative group дополнительно сохраняется `analysis/test_negative_euphemism_match_fp.md`: туда попадают только test sample-ы этой группы, на которых модель дала хотя бы один FP span. Общий `test_fp_fn.md` при этом остаётся агрегированным по всему `test`.

### 3.1. Sweep по `alpha` для `combined` head

Для короткого перебора `alpha-learning-rate` и стартового `initial-alpha` теперь есть отдельный CLI:

```bash
venv/bin/python -m src.models.sweep_alpha \
  --model-name deepvk/RuModernBERT-base \
  --epochs 3 \
  --train-batch-size 16 \
  --eval-batch-size 16 \
  --max-length 256 \
  --device cuda:1 \
  --alpha-learning-rates 1e-4 3e-4 1e-3 3e-3 1e-2 \
  --initial-alphas 0.2 0.35 0.5 0.65 0.8
```

Что делает этот запуск:

- для каждой пары `alpha-learning-rate` x `initial-alpha` запускает отдельное обучение `combined` head;
- по умолчанию использует `3` эпохи на trial;
- внутри каждого trial лучший чекпоинт выбирается по `test`;
- после завершения строит общий рейтинг запусков по `test span_f1` с tie-breaker по `test token_f1`.

Результаты sweep сохраняются в новую папку вида:

```bash
outputs/alpha_sweeps/<model>_alpha_sweep_<timestamp>/
```

Внутри сохраняются:

- `manifest.json` — параметры sweep;
- `summary.json` — полная сводка по всем trial;
- `summary.md` — короткий markdown-рейтинг лучших запусков;
- `logs/*.log` — stdout/stderr каждого training run;
- `runs/` — вложенные training run-директории со всеми обычными артефактами `src.models.train`.

Основные параметры `src.models.sweep_alpha`:

- `--alpha-learning-rates`
- `--initial-alphas`
- `--selection-metric`
- `--selection-tie-breaker-metric`
- `--top-k`
- `--fail-fast`
- `--dry-run`

### 4. Инференс на одном тексте из `txt`

Для запуска инференса на одном тексте можно использовать standalone-скрипт из корня репозитория:

```bash
venv/bin/python scripts/infer_one_text.py \
  --model-dir outputs/models/rumodernbert_base_04_24_11_04 \
  --head-mode auto \
  --input-path path/to/text.txt
```

Что поддерживается:

- `--model-dir` можно передать либо как run-директорию `outputs/models/<run_name>`, либо сразу как `outputs/models/<run_name>/best_model`;
- `--head-mode auto` читает режим из custom checkpoint, а legacy Hugging Face checkpoint считает `baseline`;
- если явно передать `--head-mode baseline`, `neighbor` или `combined`, скрипт проверит metadata checkpoint'а и остановится с понятной ошибкой при mismatch;
- `--input-path` — это обычный `txt`-файл с одним текстом; текст может занимать несколько строк;
- скрипт читает файл целиком как один input text и прогоняет его через тот же preprocessing, что и dataset pipeline;
- если текст длиннее training `max_length`, скрипт автоматически прогоняет его по перекрывающимся окнам и потом собирает итоговые token-label предсказания обратно;
- `--prediction-threshold` задаёт порог вероятности positive-класса; по умолчанию `0.5`, что соответствует прежнему `argmax`-поведению;
- в stdout печатаются найденные сущности с `char`- и `token`-offset'ами, а также текст с подсветкой `[[...]]`.

При желании можно сохранить полный результат в JSON:

```bash
venv/bin/python scripts/infer_one_text.py \
  --model-dir outputs/models/rumodernbert_base_04_24_11_04/best_model \
  --head-mode baseline \
  --input-path path/to/text.txt \
  --output-json outputs/inference/result.json
```

В JSON output дополнительно сохраняются фактические `head_mode` и `checkpoint_architecture`.

Дополнительно поддерживаются параметры:

- `--device`
- `--head-mode`
- `--max-length`
- `--window-overlap-words`
- `--prediction-threshold`
- `--print-tags`
- `--hide-highlight`

### 5. Подбор prediction threshold на synthetic split

Для подбора порога positive-класса под максимальный `span_f1` на готовом token-label split можно использовать отдельный analysis-скрипт:

```bash
venv/bin/python scripts/tune_prediction_threshold.py \
  --model-dir outputs/models/rumodernbert_base_04_29_16_37 \
  --head-mode auto \
  --device cuda:1 \
  --split test
```

Скрипт:

- принимает либо run-директорию, либо путь прямо к `best_model/`;
- по умолчанию берёт `dataset_dir`, `max_length` и `overflow_handling` из `run_config.json`;
- один раз прогоняет checkpoint по split, считает `sigmoid(positive_logit)` на word-start токенах и затем точным инкрементальным проходом перебирает candidate thresholds;
- выбирает лучший threshold по `span_f1`, с tie-breaker по `token_f1` и близости к `0.5`;
- дополнительно печатает метрики при default threshold `0.5`.
- `--log-every` управляет progress output по evaluation batch'ам; по умолчанию печатает прогресс каждые 10 batch'ей.
- перед threshold sweep скрипт печатает число candidate thresholds и word-level score-ов.

Если `--output-path` не задан, JSON сохраняется в:

```bash
outputs/models/<run_name>/threshold_tuning/test_span_f1.json
```

В JSON сохраняются checkpoint metadata, лучшая метрика, метрики при `0.5`, compact threshold curve и word-level `positive_scores` с gold tags. Если threshold подбирается на `test`, результат нужно читать как tuned-on-test analysis, а не как независимую финальную оценку.

Для быстрой проверки логики без загрузки модели:

```bash
venv/bin/python scripts/tune_prediction_threshold.py --self-test
```

### 6. Evaluation на verified real-euphemism JSON

Для отдельной проверки чекпоинта на `data/real_test_euph.json` есть CLI:

```bash
venv/bin/python -m src.evaluation \
  --model-dir outputs/models/rumodernbert_base_04_29_16_37/best_model \
  --input-json data/real_test_euph.json \
  --target-keywords-path data/target_keywords_forms_drug.txt \
  --head-mode auto \
  --output-dir outputs/evaluation/real_test_euph
```

CLI использует только записи с `verified=true`. Внутри каждой записи читаются только:

- `text_body`;
- `entities[].text`;
- `entities[].start`;
- `entities[].end`.

Лишние поля во входном JSON игнорируются. Offsets проверяются строго: для каждой entity должно выполняться `text_body[start:end] == text`.

Метрики считаются после target-keyword masking:

- target keywords загружаются из `data/target_keywords_forms_drug.txt`;
- сравнение target keyword token-ов делается через такую же нормализацию, как в data prep: lowercase, NFC, `ё -> е`;
- если target keyword встречается в тексте, размечен как entity или предсказан моделью, соответствующий token принудительно считается `O` и в gold, и в pred;
- такие token-ы не дают ни TP, ни FP, ни FN для token/span precision, recall и F1.

В `--output-dir` сохраняются:

- `metrics.json` — checkpoint metadata, counts и token/span precision, recall, F1;
- `predictions.jsonl` — raw и masked gold/pred tags, ignored token positions и span-ы;
- `analysis/fp_fn.md` — человеко-читаемый FP/FN отчёт по masked-разметке.

## Что уже реализовано

- загрузка и очистка raw texts;
- поиск канонических форм по словарю;
- морфологически согласованная подстановка эвфемизмов;
- подготовка data split dataset с positives и negatives;
- преобразование split JSON в token-label датасет;
- training loop для `ModernBERT` / `RuModernBERT` в `src.models`;
- custom binary heads: `baseline`, `neighbor`, `combined`;
- `word_start_mask` для train/eval/inference;
- checkpoint loading с custom/legacy compatibility checks;
- token-level и span-level evaluation;
- отдельный CLI для подбора prediction threshold по `span_f1` на готовом split;
- отдельный CLI для evaluation на verified real-euphemism JSON с ignore target-keyword masking;
- сохранение лучшего чекпоинта, predictions и metrics;
- отдельный читаемый `test`-лог FP/FN для span-level error analysis;
- reproducible sampling и split по `seed` на этапе подготовки данных;
- предупреждения в терминал, если char-level span нечётко совпадает с границами токенов;
- `run_config.json` для воспроизводимого training run.

## Что пока не реализовано

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

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Iterable

from src.data.io import load_lines, write_json
from src.data.text import WORD_RE, lookup_norm, nfc
from src.synthetic.morphology import inflect_like, normalize_euphemism_lemma
from src.synthetic.targets import build_target_forms


@dataclass(frozen=True)
class EuphemismAnnotation:
    target_word: str
    target_lemma: str
    base_euphemism: str
    euphemism: str
    source_start: int
    source_end: int
    start: int
    end: int


@dataclass(frozen=True)
class SyntheticSample:
    text: str
    replaced_text: str
    euphemisms: list[EuphemismAnnotation]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["euphemisms"] = [asdict(item) for item in self.euphemisms]
        return payload


def choose_euphemism_lemma(
    euphemisms: list[str],
    used_in_text: set[str],
    rng: random.Random,
) -> str:
    available = [euphemism for euphemism in euphemisms if euphemism not in used_in_text]
    if available:
        return rng.choice(available)
    return rng.choice(euphemisms)


def replace_in_text(
    text: str,
    *,
    form_to_lemma: dict[str, str],
    euphemisms: list[str],
    rng: random.Random,
) -> SyntheticSample:
    normalized_text = nfc(text)
    lemma_to_euphemism: dict[str, str] = {}
    used_euphemisms_in_text: set[str] = set()

    replaced_parts: list[str] = []
    annotations: list[EuphemismAnnotation] = []

    last_pos = 0
    replaced_cursor = 0

    for match in WORD_RE.finditer(normalized_text):
        start, end = match.span()
        token = match.group(0)

        prefix = normalized_text[last_pos:start]
        replaced_parts.append(prefix)
        replaced_cursor += len(prefix)

        token_norm = lookup_norm(token)
        target_lemma = form_to_lemma.get(token_norm)

        if target_lemma is None:
            replaced_parts.append(token)
            replaced_cursor += len(token)
            last_pos = end
            continue

        if target_lemma not in lemma_to_euphemism:
            base_euphemism = choose_euphemism_lemma(
                euphemisms,
                used_in_text=used_euphemisms_in_text,
                rng=rng,
            )
            lemma_to_euphemism[target_lemma] = base_euphemism
            used_euphemisms_in_text.add(base_euphemism)
        else:
            base_euphemism = lemma_to_euphemism[target_lemma]

        replacement = inflect_like(base_euphemism, token, target_lemma)
        replacement_start = replaced_cursor
        replacement_end = replacement_start + len(replacement)

        replaced_parts.append(replacement)
        replaced_cursor = replacement_end

        annotations.append(
            EuphemismAnnotation(
                target_word=token,
                target_lemma=target_lemma,
                base_euphemism=base_euphemism,
                euphemism=replacement,
                source_start=start,
                source_end=end,
                start=replacement_start,
                end=replacement_end,
            )
        )
        last_pos = end

    tail = normalized_text[last_pos:]
    replaced_parts.append(tail)

    return SyntheticSample(
        text=normalized_text,
        replaced_text="".join(replaced_parts),
        euphemisms=annotations,
    )


def load_euphemism_lemmas(path: str) -> list[str]:
    raw_euphemisms = load_lines(path)
    normalized = [normalize_euphemism_lemma(word) for word in raw_euphemisms]
    deduplicated = list(dict.fromkeys(normalized))
    if not deduplicated:
        raise ValueError("Euphemism vocabulary is empty.")
    return deduplicated


def build_synthetic_samples(
    texts: Iterable[str],
    *,
    form_to_lemma: dict[str, str],
    euphemisms: list[str],
    seed: int | None = 42,
) -> list[SyntheticSample]:
    rng = random.Random(seed)
    return [
        replace_in_text(
            text,
            form_to_lemma=form_to_lemma,
            euphemisms=euphemisms,
            rng=rng,
        )
        for text in texts
    ]


def build_data_json(
    *,
    texts_path: str = "data/drug_texts_small.txt",
    euphemisms_path: str = "data/real_euphemisms.txt",
    target_words_path: str = "data/target_keywords_forms_drug.txt",
    output_path: str = "outputs/synthetic/data.json",
    seed: int | None = 42,
) -> list[dict]:
    texts = load_lines(texts_path)
    euphemisms = load_euphemism_lemmas(euphemisms_path)
    target_words = load_lines(target_words_path)
    form_to_lemma = build_target_forms(target_words)

    samples = build_synthetic_samples(
        texts,
        form_to_lemma=form_to_lemma,
        euphemisms=euphemisms,
        seed=seed,
    )
    payload = [sample.to_dict() for sample in samples if sample.euphemisms]
    write_json(output_path, payload)
    return payload

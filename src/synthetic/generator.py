from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Iterable

from src.data.io import load_lines, write_json
from src.data.text import WORD_RE, lookup_norm, nfc
from src.synthetic.morphology import (
    can_inflect_to_plural,
    get_word_number,
    inflect_like,
)
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


@dataclass(frozen=True)
class EuphemismEntry:
    text: str
    number: str | None
    can_be_pluralized: bool


def is_number_compatible(
    euphemism: EuphemismEntry,
    *,
    target_number: str | None,
) -> bool:
    if target_number == "sing":
        return euphemism.number != "plur"
    if target_number == "plur":
        return euphemism.number == "plur" or euphemism.can_be_pluralized
    return True


def is_replacement_number_compatible(
    replacement: str,
    *,
    target_number: str | None,
) -> bool:
    replacement_number = get_word_number(replacement)
    if target_number == "sing":
        return replacement_number != "plur"
    if target_number == "plur":
        return replacement_number == "plur"
    return True


def choose_euphemism(
    euphemisms: list[EuphemismEntry],
    *,
    target_word: str,
    target_lemma: str,
    target_number: str | None,
    used_in_text: set[str],
    rng: random.Random,
) -> tuple[EuphemismEntry, str]:
    compatible = [
        euphemism
        for euphemism in euphemisms
        if is_number_compatible(euphemism, target_number=target_number)
    ]
    if not compatible:
        raise ValueError(
            f"No number-compatible euphemism found for target number {target_number!r}."
        )

    available = [
        euphemism for euphemism in compatible if euphemism.text not in used_in_text
    ]

    for pool in (available, compatible):
        if not pool:
            continue

        shuffled = pool[:]
        rng.shuffle(shuffled)
        for euphemism in shuffled:
            replacement = inflect_like(euphemism.text, target_word, target_lemma)
            if is_replacement_number_compatible(
                replacement,
                target_number=target_number,
            ):
                return euphemism, replacement

    raise ValueError(
        f"No euphemism produced a number-compatible replacement for {target_word!r}."
    )


def replace_in_text(
    text: str,
    *,
    form_to_lemma: dict[str, str],
    euphemisms: list[EuphemismEntry],
    rng: random.Random,
) -> SyntheticSample:
    normalized_text = nfc(text)
    lemma_to_euphemism: dict[str, EuphemismEntry] = {}
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

        target_number = get_word_number(token, expected_lemma=target_lemma)
        selected_euphemism = lemma_to_euphemism.get(target_lemma)
        replacement = None

        if selected_euphemism is not None and is_number_compatible(
            selected_euphemism,
            target_number=target_number,
        ):
            candidate_replacement = inflect_like(
                selected_euphemism.text,
                token,
                target_lemma,
            )
            if is_replacement_number_compatible(
                candidate_replacement,
                target_number=target_number,
            ):
                replacement = candidate_replacement

        if replacement is None:
            selected_euphemism, replacement = choose_euphemism(
                euphemisms,
                target_word=token,
                target_lemma=target_lemma,
                target_number=target_number,
                used_in_text=used_euphemisms_in_text,
                rng=rng,
            )
            lemma_to_euphemism[target_lemma] = selected_euphemism
            used_euphemisms_in_text.add(selected_euphemism.text)
        replacement_start = replaced_cursor
        replacement_end = replacement_start + len(replacement)

        replaced_parts.append(replacement)
        replaced_cursor = replacement_end

        annotations.append(
            EuphemismAnnotation(
                target_word=token,
                target_lemma=target_lemma,
                base_euphemism=selected_euphemism.text,
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


def load_euphemisms(path: str) -> list[EuphemismEntry]:
    deduplicated = list(dict.fromkeys(load_lines(path)))
    if not deduplicated:
        raise ValueError("Euphemism vocabulary is empty.")
    return [
        EuphemismEntry(
            text=euphemism,
            number=get_word_number(euphemism),
            can_be_pluralized=can_inflect_to_plural(euphemism),
        )
        for euphemism in deduplicated
    ]


def build_synthetic_samples(
    texts: Iterable[str],
    *,
    form_to_lemma: dict[str, str],
    euphemisms: list[EuphemismEntry],
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
    euphemisms = load_euphemisms(euphemisms_path)
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

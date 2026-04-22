from __future__ import annotations

from src.data.text import lookup_norm
from src.data_prep.morphology import require_morph


def build_target_forms(target_words: list[str]) -> dict[str, str]:
    morph = require_morph()
    form_to_lemma: dict[str, str] = {}

    for form in target_words:
        normalized_form = lookup_norm(form)
        parses = morph.parse(form)
        lemma = parses[0].normal_form if parses else normalized_form
        form_to_lemma[normalized_form] = lemma

    return form_to_lemma


def build_observed_euphemism_forms(
    euphemisms: list[str],
) -> dict[str, tuple[str, str]]:
    morph = require_morph()
    form_to_base_and_lemma: dict[str, tuple[str, str]] = {}

    for euphemism in euphemisms:
        normalized_euphemism = lookup_norm(euphemism)
        parses = morph.parse(euphemism)
        if not parses:
            form_to_base_and_lemma.setdefault(
                normalized_euphemism,
                (euphemism, normalized_euphemism),
            )
            continue

        base_parse = next(
            (parse for parse in parses if lookup_norm(parse.word) == normalized_euphemism),
            parses[0],
        )
        lemma = base_parse.normal_form or normalized_euphemism

        related_parses = [
            parse for parse in parses if parse.normal_form == base_parse.normal_form
        ] or [base_parse]

        candidate_forms = {normalized_euphemism}
        for parse in related_parses:
            candidate_forms.add(lookup_norm(parse.word))
            try:
                candidate_forms.update(lookup_norm(form.word) for form in parse.lexeme)
            except Exception:  # pragma: no cover - defensive around pymorphy internals
                continue

        for normalized_form in candidate_forms:
            form_to_base_and_lemma.setdefault(
                normalized_form,
                (euphemism, lemma),
            )

    return form_to_base_and_lemma

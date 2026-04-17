from __future__ import annotations

from src.data.text import lookup_norm
from src.synthetic.morphology import require_morph


def build_target_forms(target_words: list[str]) -> dict[str, str]:
    morph = require_morph()
    form_to_lemma: dict[str, str] = {}

    for form in target_words:
        normalized_form = lookup_norm(form)
        parses = morph.parse(form)
        lemma = parses[0].normal_form if parses else normalized_form
        form_to_lemma[normalized_form] = lemma

    return form_to_lemma

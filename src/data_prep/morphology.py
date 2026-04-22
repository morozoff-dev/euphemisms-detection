from __future__ import annotations

from functools import lru_cache

from src.data.text import preserve_case

try:
    from pymorphy3 import MorphAnalyzer
except ImportError:  # pragma: no cover - depends on local environment
    MorphAnalyzer = None


INFLECT_GRAMMEMES = {
    "nomn",
    "gent",
    "datv",
    "accs",
    "ablt",
    "loct",
    "voct",
    "gen2",
    "acc2",
    "loc2",
    "sing",
    "plur",
    "masc",
    "femn",
    "neut",
}


def require_morph():
    if MorphAnalyzer is None:
        raise RuntimeError(
            "pymorphy3 is required for data preparation. "
            "Install dependencies from requirements.txt first."
        )
    return get_morph()


@lru_cache(maxsize=1)
def get_morph():
    if MorphAnalyzer is None:
        raise RuntimeError(
            "pymorphy3 is required for data preparation. "
            "Install dependencies from requirements.txt first."
        )
    return MorphAnalyzer()


def best_parse(
    word: str,
    *,
    expected_lemma: str | None = None,
    desired_pos: str | None = None,
):
    morph = require_morph()
    parses = morph.parse(word)
    if not parses:
        return None

    scored = []
    for parse in parses:
        score = 0
        if expected_lemma and parse.normal_form == expected_lemma:
            score += 10
        if desired_pos and parse.tag.POS == desired_pos:
            score += 5
        scored.append((score, parse.score, parse))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def get_word_number(
    word: str,
    *,
    expected_lemma: str | None = None,
    desired_pos: str | None = None,
) -> str | None:
    parse = best_parse(
        word,
        expected_lemma=expected_lemma,
        desired_pos=desired_pos,
    )
    if parse is None:
        return None
    return parse.tag.number


def can_inflect_to_plural(word: str) -> bool:
    parse = best_parse(word)
    if parse is None:
        return False
    if parse.tag.number == "plur":
        return True
    return parse.inflect({"plur"}) is not None


def inflect_like(base_euphemism: str, target_word: str, target_lemma: str) -> str:
    morph = require_morph()
    target_parse = best_parse(target_word, expected_lemma=target_lemma)
    if target_parse is None:
        return preserve_case(target_word, base_euphemism)

    euphemism_parse = best_parse(base_euphemism, desired_pos=target_parse.tag.POS)
    if euphemism_parse is None:
        euphemism_parse = morph.parse(base_euphemism)[0]

    grammemes = {
        grammeme
        for grammeme in target_parse.tag.grammemes
        if grammeme in INFLECT_GRAMMEMES
    }
    inflected = euphemism_parse.inflect(grammemes) if grammemes else None
    if inflected:
        return preserve_case(target_word, inflected.word)

    fallback_grammemes = set()
    if target_parse.tag.case:
        fallback_grammemes.add(target_parse.tag.case)
    if target_parse.tag.number:
        fallback_grammemes.add(target_parse.tag.number)

    inflected = (
        euphemism_parse.inflect(fallback_grammemes) if fallback_grammemes else None
    )
    if inflected:
        return preserve_case(target_word, inflected.word)

    return preserve_case(target_word, base_euphemism)

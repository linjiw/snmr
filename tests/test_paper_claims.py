"""Pin the manuscript's prose to the properties the code tests actually establish.

The shuffled arm (S) is a phase-matched MISALIGNED-REFERENCE control. Its donor map
is a deterministic bijection, so clip identity survives it at every pool size --
see docs/E70_SHUFFLE_CONTROL_AUDIT_2026-09-10.md and the probes in
tests/test_distillation.py. Describing it as identity-erasing or content-free is a
factual error about the frozen experiment, and it was in the manuscript until
2026-09-10. This guard stops it coming back during an edit under deadline pressure.
"""

from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1] / "paper" / "main.tex"

# Phrases that assert a property the shuffle control does not have.
BANNED = (
    "identity-erasing",
    "content-free null",
    "destroys clip identity",
    "erases clip identity",
    "destroy clip identity",
    "wrong-trajectory control",
)


# A negated mention ("not an identity-erasing one") is the correct usage, so only
# ASSERTED occurrences count. Anything within this many characters after a negator
# is treated as a denial rather than a claim.
NEGATORS = ("not an ", "not a ", "not ", "neither ", "rather than an ", "rather than ")
NEGATION_WINDOW = 24


def _asserted_occurrences(text: str, phrase: str) -> list[int]:
    hits, start = [], 0
    while (index := text.find(phrase, start)) != -1:
        prefix = text[max(0, index - NEGATION_WINDOW) : index]
        if not any(prefix.rstrip().endswith(n.strip()) for n in NEGATORS):
            hits.append(index)
        start = index + len(phrase)
    return hits


@pytest.mark.skipif(not PAPER.is_file(), reason="manuscript not present")
@pytest.mark.parametrize("phrase", BANNED)
def test_manuscript_does_not_claim_the_shuffle_erases_identity(phrase):
    text = PAPER.read_text(encoding="utf-8").lower()
    hits = _asserted_occurrences(text, phrase)
    context = [text[max(0, i - 70) : i + len(phrase) + 20] for i in hits]
    assert not hits, (
        f"paper/main.tex asserts {phrase!r} at {hits}. The S arm withholds the correct "
        "future; it does not remove clip identity. The identity-free null is the time "
        f"code (T).\nContext: {context}"
    )


def test_the_negation_guard_itself_works():
    """A denial must pass and a bare assertion must fail, or the guard is decorative."""
    assert _asserted_occurrences("it is not an identity-erasing control", "identity-erasing") == []
    assert _asserted_occurrences("the identity-erasing control", "identity-erasing") != []


@pytest.mark.skipif(not PAPER.is_file(), reason="manuscript not present")
def test_manuscript_names_the_time_code_as_the_identity_free_null():
    """The positive half: the paper must say which arm actually is identity-free."""
    text = PAPER.read_text(encoding="utf-8")
    assert "identity-free" in text, (
        "The manuscript should state explicitly that T is the identity-free null, "
        "since S is not."
    )

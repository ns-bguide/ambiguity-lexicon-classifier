import math
from types import SimpleNamespace

from ambiguity.scorer import AmbiguityModelEN


class FakeResources:
    def __init__(self, entries):
        self._entries = entries

    def get_entry(self, term):
        return self._entries.get(term)

    def freq(self, term):
        e = self._entries.get(term)
        return 0.0 if e is None else e.freq_wordfreq


def make_entry(token: str, freq: float = 0.0, n_lexicons: int = 0, in_wordfreq: bool = False, in_hunspell: bool = False, in_wiktionary: bool = False):
    return SimpleNamespace(
        token=token,
        len_token=len(token),
        lemma=token,
        freq_wordfreq=freq,
        in_hunspell=in_hunspell,
        in_wiktionary=in_wiktionary,
        in_wordfreq=in_wordfreq,
        in_extra_sources=False,
        n_lexicons=n_lexicons,
        pos_set=None,
        extra=None,
    )


def test_ambiguous_common_word():
    entries = {
        "bank": make_entry("bank", freq=1e-3, n_lexicons=3, in_wordfreq=True, in_wiktionary=True),
    }
    model = AmbiguityModelEN(FakeResources(entries))
    res = model.score_terms(["bank"])[0]
    assert res["label"] == "likely ambiguous"
    assert res["debug"]["n_lexicons"] >= 2
    assert res["debug"]["freq_log"] >= model.freq_log_common or res["debug"]["in_wordfreq"]


def test_unambiguous_long_or_rare():
    # Very long technical term
    entries = {
        "pneumonoultramicroscopicsilicovolcanoconiosis": make_entry(
            "pneumonoultramicroscopicsilicovolcanoconiosis", freq=1e-12, n_lexicons=1, in_wiktionary=True
        )
    }
    model = AmbiguityModelEN(FakeResources(entries))
    res = model.score_terms(["pneumonoultramicroscopicsilicovolcanoconiosis"])[0]
    assert res["label"] in {"likely unambiguous", "need review"}
    # Now force unambiguous via thresholds
    model2 = AmbiguityModelEN(FakeResources(entries), thresholds={"long_token_len": 20, "freq_log_rare": -10.0})
    res2 = model2.score_terms(["pneumonoultramicroscopicsilicovolcanoconiosis"])[0]
    assert res2["label"] == "likely unambiguous"


def test_threshold_overrides_affect_label():
    entries = {
        "zebra": make_entry("zebra", freq=1e-8, n_lexicons=1, in_wiktionary=True),
    }
    # With stricter common threshold, zebra should be review/unambiguous
    model = AmbiguityModelEN(FakeResources(entries), thresholds={"freq_log_common": -12.0})
    res = model.score_terms(["zebra"])[0]
    assert res["label"] in {"likely unambiguous", "need review"}
    # With len threshold low, it could sway to review
    model2 = AmbiguityModelEN(FakeResources(entries), thresholds={"long_token_len": 3})
    res2 = model2.score_terms(["zebra"])[0]
    assert res2["label"] in {"need review", "likely unambiguous"}

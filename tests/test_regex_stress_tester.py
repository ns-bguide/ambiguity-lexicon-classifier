from regex_tester.stress_tester import RegexStressTester


def test_search_mode_matches_across_languages():
    token_sources = {
        "en": ["cat", "catalog", "dog", "Bank"],
        "pt": ["casa", "catalogo", "banco"],
    }
    tester = RegexStressTester(token_sources=token_sources)
    stats, matches = tester.run(pattern="cat", languages=["en", "pt"], mode="search")
    assert matches["en"] == ["cat", "catalog"]
    assert matches["pt"] == ["catalogo"]
    assert any(s.lang == "en" and s.matched == 2 for s in stats)
    assert any(s.lang == "pt" and s.matched == 1 for s in stats)


def test_fullmatch_and_ignore_case():
    token_sources = {
        "en": ["Bank", "banking", "BANK"],
        "pt": ["banco", "Banco"],
    }
    tester = RegexStressTester(token_sources=token_sources)
    stats, matches = tester.run(pattern=r"bank", languages=["en", "pt"], mode="fullmatch", ignore_case=True)
    assert set(matches["en"]) == {"Bank", "BANK"}
    assert set(matches["pt"]) == set()  # 'banco'/'Banco' are not exact 'bank'


def test_limits_and_timeouts_do_not_error():
    token_sources = {"en": [str(i) for i in range(10000)]}
    tester = RegexStressTester(token_sources=token_sources)
    stats, matches = tester.run(pattern=r"^9999$", languages=["en"], mode="fullmatch", limit_per_lang=1000, timeout_sec_per_lang=0.01)
    # Not asserting specific counts due to timing variance; ensure types and keys exist
    assert "en" in matches
    assert len(stats) == 1

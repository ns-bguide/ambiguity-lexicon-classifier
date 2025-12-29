from regex_tester.stress_tester import RegexStressTester


def test_include_metadata_with_token_sources_falls_back_len_only():
    token_sources = {
        "en": ["alpha", "beta", "gamma"],
    }
    tester = RegexStressTester(token_sources=token_sources)
    stats, matches = tester.run(pattern=r"a", languages=["en"], mode="search", include_metadata=True, metadata_fields=["len_token", "n_lexicons"])  # noqa: E501
    items = matches["en"]
    assert isinstance(items[0], dict)
    # token present
    assert all("token" in it for it in items)
    # len_token filled, n_lexicons unknown => None
    assert any(it.get("len_token") == len(it["token"]) for it in items)
    assert all("n_lexicons" in it for it in items)

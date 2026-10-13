import numpy as np
import pandas as pd
import pytest

from capalloc.experiment import paired_cluster_test, split, validation_interval, load_panel


def test_split_uses_label_publication_and_prediction_dates():
    d = pd.DataFrame({"id": list("abcdef"),
        "known_at": ["2020-01-01", "2021-11-01", "2022-02-01", "2022-03-01", "2024-01-01", "2025-02-01"],
        "target_known_at": ["2021-01-01", "2022-11-01", "2023-01-01", "2024-03-01", "2025-01-01", "2026-01-01"]})
    tr, va, te = split(d)
    assert tr.id.tolist() == ["a"]
    assert va.id.tolist() == ["c"]
    assert te.id.tolist() == ["e", "f"]
    assert tr.target_known_at.max() < va.known_at.min()
    assert va.target_known_at.max() < te.known_at.min()


def test_paired_cluster_comparison_respects_matching_and_sign():
    y = np.zeros(8)
    base = np.ones(8)
    candidate = np.zeros(8)
    companies = np.array(["a"] * 4 + ["b"] * 4)
    result = paired_cluster_test(np.arange(8), companies, y, base, candidate, repeats=200)
    assert result["improvement_pp"] == 100
    assert result["n_companies"] == 2
    assert result["cluster_bootstrap_95_ci_pp"] == [100., 100.]
    with pytest.raises(ValueError, match="unique IDs"):
        paired_cluster_test([1, 1], ["a", "a"], [0, 0], [1, 1], [0, 0])


def test_validation_interval_uses_finite_sample_rank():
    lo, hi, q = validation_interval(np.arange(19), np.zeros(19), np.array([0.]), .1)
    assert q == 17 and hi[0] == 17 and lo[0] == -17


def test_annual_lag_requires_adjacent_year_and_earlier_filing(tmp_path):
    p = tmp_path / "panel.csv"
    pd.DataFrame({"id": ["1:2019", "1:2020", "1:2022"], "cik": ["1"] * 3,
        "fiscal_end": ["2019-12-31", "2020-12-31", "2022-12-31"],
        "known_at": ["2020-02-01", "2021-02-01", "2023-02-01"],
        "target_known_at": ["2021-02-01", "2022-02-01", "2024-02-01"],
        "revenue": [100, 110, 130], "capex_rate": [.1, .2, .3], "repurchase_rate": [.1, .1, .1]}).to_csv(p, index=False)
    d = load_panel(p).set_index("id")
    assert d.loc["1:2020", "revenue_growth"] == pytest.approx(.1)
    assert np.isnan(d.loc["1:2022", "revenue_growth"])

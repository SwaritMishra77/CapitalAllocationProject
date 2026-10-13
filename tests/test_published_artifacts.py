"""Published row predictions must agree with the visible score table."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_published_scores_and_paired_contrast_agree_with_rows():
    metrics = json.loads((ROOT / "reports/metrics.json").read_text())
    predictions = pd.read_csv(ROOT / "reports/test_predictions.csv", dtype={"cik": str})
    for cohort, group in predictions.groupby("cohort"):
        assert group.id.is_unique
        expected = metrics["cohorts"][cohort]
        assert len(group) == expected["counts"]["test"]
        assert group.cik.nunique() == expected["counts"]["test_companies"]
        assert (group.known_at < group.target_known_at).all()
        for model, score in expected["models"].items():
            observed = np.mean(np.abs(group.actual - group[model])) * 100
            assert np.isclose(observed, score["mae_pp"], atol=1e-10), model
    matched = predictions[predictions.cohort == "text"]
    base = metrics["hypothesis_test"]["baseline"]
    difference = (np.abs(matched.actual - matched[base]) -
                  np.abs(matched.actual - matched.numeric_plus_transformer)).mean() * 100
    assert np.isclose(difference, metrics["hypothesis_test"]["improvement_pp"], atol=1e-10)
    assert metrics["hypothesis_test"]["n_rows"] == len(matched)
    assert metrics["hypothesis_test"]["n_companies"] == matched.cik.nunique()
    assert set(matched.id).issubset(set(predictions[predictions.cohort == "all"].id))

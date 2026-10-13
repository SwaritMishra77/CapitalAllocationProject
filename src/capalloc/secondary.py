"""Descriptive secondary outcomes with observed XBRL labels only."""

import argparse
import json
from pathlib import Path

from .experiment import evaluate, fit_residual_ridge, load_panel, matrix, numeric_preprocessor, split


def run(panel, output):
    d = load_panel(panel)
    results = {"description": "secondary, descriptive forecasts; missing tags are excluded, never coded as zero"}
    for category in ("repurchase", "dividend", "rnd"):
        source = category + "_rate"
        target = "next_" + category
        sample = d[d[source].notna() & d[target].notna()].copy()
        sample["outcome"] = sample[target] / sample.next_revenue
        train, valid, test = split(sample)
        if min(len(train), len(valid), len(test)) < 8:
            results[category] = {"status": "insufficient observed labels", "counts": [len(train), len(valid), len(test)]}
            continue
        a, b, c = matrix(numeric_preprocessor(), train, valid, test)
        va, te, choice = fit_residual_ridge(a, train.outcome.to_numpy(), b, valid.outcome.to_numpy(), c,
                                            train[source].to_numpy(), valid[source].to_numpy(), test[source].to_numpy())
        y = test.outcome.to_numpy()
        results[category] = {"counts": {"train": len(train), "validation": len(valid), "test": len(test),
                                         "test_companies": test.cik.nunique()},
                             "persistence": evaluate(y, test[source].to_numpy()),
                             "numeric_residual_ridge": {**evaluate(y, te), "validation_choice": choice}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, default=Path("data/processed/panel.csv"))
    p.add_argument("--out", type=Path, default=Path("reports/secondary.json"))
    a = p.parse_args()
    print(json.dumps(run(a.panel, a.out), indent=2))


if __name__ == "__main__":
    main()

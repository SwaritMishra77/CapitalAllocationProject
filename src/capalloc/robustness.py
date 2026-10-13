"""A supplementary unseen-company numerical test, with a frozen CIK split."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error

from .experiment import (evaluate, fit_residual_ridge, load_panel, matrix,
                         numeric_preprocessor, split)


def holdout_ciks(panel, per_sector=10):
    companies = panel[["cik", "sector"]].drop_duplicates()
    selected = []
    for sector, group in companies.groupby("sector"):
        chosen = group.assign(key=group.cik.map(lambda x: hashlib.sha256(x.encode()).hexdigest())).sort_values("key").head(per_sector)
        selected.extend(chosen.cik.tolist())
    return sorted(selected)


def run(panel, output):
    d = load_panel(panel)
    held = holdout_ciks(d)
    train, valid, test = split(d)
    train, valid, test = train[~train.cik.isin(held)], valid[~valid.cik.isin(held)], test[test.cik.isin(held)]
    a, b, c = matrix(numeric_preprocessor(), train, valid, test)
    y, yv, yt = [x.target_capex_rate.to_numpy() for x in (train, valid, test)]
    persistence = test.capex_rate.to_numpy()
    vv, pp, choice = fit_residual_ridge(a, y, b, yv, c, train.capex_rate.to_numpy(),
                                         valid.capex_rate.to_numpy(), persistence)
    result = {"description": "unseen-company numerical robustness, not the primary text hypothesis",
              "rule": "SHA-256 CIK sort, first 10 in each of retail and manufacturing",
              "heldout_ciks": held, "counts": {"train": len(train), "validation": len(valid), "test": len(test),
                                          "test_companies": test.cik.nunique()},
              "persistence": evaluate(yt, persistence), "numeric_residual_ridge": evaluate(yt, pp),
              "ridge_validation_choice": choice}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, default=Path("data/processed/panel.csv"))
    p.add_argument("--out", type=Path, default=Path("reports/unseen_company.json"))
    args = p.parse_args()
    print(json.dumps(run(args.panel, args.out), indent=2))


if __name__ == "__main__":
    main()

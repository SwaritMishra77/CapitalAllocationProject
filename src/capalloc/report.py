"""Render recorded measurements and protocol caveats from the experiment artifacts."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .experiment import split


def make_report(metrics, predictions, panel, unseen, secondary, output):
    m = json.loads(Path(metrics).read_text())
    p = pd.read_csv(predictions, dtype={"cik": str})
    d = pd.read_csv(panel, dtype={"cik": str})
    u = json.loads(Path(unseen).read_text())
    sec = json.loads(Path(secondary).read_text())
    h = m["hypothesis_test"]
    b = h["baseline"]
    bm = m["benchmark_comparison"]
    all_models, text_models = [m["cohorts"][c]["models"] for c in ("all", "text")]
    n_all, n_text = [m["cohorts"][c]["counts"] for c in ("all", "text")]
    a = m["cohorts"]["all"]["availability_audit"]
    t = m["cohorts"]["text"]["availability_audit"]
    ci = h["cluster_bootstrap_95_ci_pp"]
    lines = ["# Results", "", "## Primary paired comparison", "",
        f"On **{h['n_rows']} test filings from {h['n_companies']} companies with cached text**, the numeric residual ridge had {text_models[b]['mae_pp']:.3f} percentage points (pp) mean absolute error (MAE). The same numeric model with frozen FinBERT embeddings had {text_models['numeric_plus_transformer']['mae_pp']:.3f} pp MAE. Mean paired improvement was **{h['improvement_pp']:+.3f} pp** (positive favors text); company-cluster bootstrap 95% interval **[{ci[0]:+.3f}, {ci[1]:+.3f}] pp**; two-sided company sign-flip p = **{h['two_sided_cluster_sign_flip_p']:.3f}**. The interval includes zero. This sample does not establish a reliable benefit from the transformer.", "",
        f"Against the validation-selected numeric benchmark ({bm['baseline']}), the same paired comparison was {bm['improvement_pp']:+.3f} pp (95% cluster interval [{bm['cluster_bootstrap_95_ci_pp'][0]:+.3f}, {bm['cluster_bootstrap_95_ci_pp'][1]:+.3f}] pp). This is a distinct benchmark, selected on validation rather than for the fixed ablation. Both are exploratory: earlier test results were inspected before this availability audit and four-chunk revision. No confirmatory p-value is claimed.", "",
        f"The 100-company source universe produced {n_all['train']} training, {n_all['validation']} validation and {n_all['test']} held-out numerical rows ({n_all['test_companies']} test firms). Cached filing text is available for only 24 source firms; the matched text experiment has {n_text['train']}/{n_text['validation']}/{n_text['test']} rows and {n_text['test_companies']} test firms. These are not random text coverage or financial returns.", "",
        "## Publication-date gate", "",
        "Forecast features are available at the source 10-K filing date. Targets enter training or tuning only after the *next* original 10-K was filed. These gates prevent later outcomes from entering an earlier decision:", "",
        "| Cohort | Latest train label | First validation forecast | Latest validation label | First test forecast |", "| --- | --- | --- | --- | --- |",
        f"| All | {a['latest_training_label']} | {a['earliest_validation_prediction']} | {a['latest_validation_label']} | {a['earliest_test_prediction']} |",
        f"| Text | {t['latest_training_label']} | {t['earliest_validation_prediction']} | {t['latest_validation_label']} | {t['earliest_test_prediction']} |", "",
        "The prior fiscal-year split in the first release allowed labels that would not yet have been published. Its scores and sample sizes have been superseded by this revision. See [audit notes](AUDIT.md).", "",
        "## Held-out forecast error", "", "MAE is measured in percentage points of next-year capital expenditure divided by next-year revenue. All choices and preprocessing use earlier available data.", "",
        "| Model | Full cohort MAE (pp) | Text-matched MAE (pp) |", "| --- | ---: | ---: |"]
    names = ["persistence", "sector_median", "numeric_ridge", "numeric_residual_ridge", "numeric_tree", "numeric_residual_tree", "numeric_gru", "text_tfidf", "numeric_plus_tfidf", "transformer_text", "numeric_plus_transformer"]
    for name in names:
        va, vb = all_models.get(name, {}).get("mae_pp"), text_models.get(name, {}).get("mae_pp")
        if va is not None or vb is not None:
            lines.append(f"| {name.replace('_', ' ')} | {va:.3f} | {vb:.3f} |" if va is not None and vb is not None else
                         f"| {name.replace('_', ' ')} | {va:.3f} | — |" if va is not None else
                         f"| {name.replace('_', ' ')} | — | {vb:.3f} |")
    interval = all_models["numeric_ridge"]["interval_90"]
    lines += ["", f"The numeric ridge 90% **validation residual diagnostic** covered {interval['coverage']:.1%} of test rows with mean width {interval['mean_width_pp']:.2f} pp. Validation residuals also chose model settings, and time distributions can shift; this interval has no formal 90% coverage guarantee.", "",
        "## Matched subgroup descriptions", "", f"Every subgroup compares {b} and numeric-plus-transformer on the same rows. Small groups are descriptive only.", "",
        "| Segment | Rows | Numbers MAE (pp) | Numbers + text MAE (pp) |", "| --- | ---: | ---: | ---: |"]
    text = p[p.cohort == "text"].merge(d[["id", "revenue"]], on="id", validate="one_to_one")
    median = split(d)[0].revenue.median()
    text["size"] = text.revenue.map(lambda z: "Below train median revenue" if z < median else "At/above train median revenue")
    for col in ["sector", "fiscal_end", "size"]:
        for label, group in text.groupby(text[col].str[:4] if col == "fiscal_end" else col):
            x = (group.actual - group[b]).abs().mean() * 100
            y = (group.actual - group.numeric_plus_transformer).abs().mean() * 100
            lines.append(f"| {label} | {len(group)} | {x:.3f} | {y:.3f} |")
    full = p[p.cohort == "all"].copy()
    full["abs_error"] = (full.actual - full.numeric_residual_ridge).abs() * 100
    lines += ["", "## Largest numerical errors", "", "These observations warrant filing review; model error is not an analyst alert.", "",
        "| Company | Source fiscal year | Actual (pp) | Forecast (pp) | Error (pp) |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in full.nlargest(5, "abs_error").itertuples():
        lines.append(f"| {row.ticker} | {row.fiscal_end[:4]} | {row.actual*100:.2f} | {row.numeric_residual_ridge*100:.2f} | {row.abs_error:.2f} |")
    lines += ["", "## Unseen-company check", "",
        f"A fixed hash-based split kept 20 companies out of model fitting. Its {u['counts']['test']} test rows ({u['counts']['test_companies']} firms) gave persistence MAE {u['persistence']['mae_pp']:.3f} pp and numerical residual ridge MAE {u['numeric_residual_ridge']['mae_pp']:.3f} pp. This is a numerical portability check only.", "",
        "## Other reported spending", "", "Secondary targets require both source and next-year tags to be observed. A missing tag is never assumed to be zero. R&D is an expense and is analyzed separately from capital cash spending.", "",
        "| Next-year ratio | Test rows | Persistence MAE (pp) | Numerical residual ridge MAE (pp) |", "| --- | ---: | ---: | ---: |"]
    for name in ("repurchase", "dividend", "rnd"):
        result = sec[name]
        if isinstance(result.get("counts"), dict):
            lines.append(f"| {name} / revenue | {result['counts']['test']} | {result['persistence']['mae_pp']:.3f} | {result['numeric_residual_ridge']['mae_pp']:.3f} |")
    lines += ["", "## Reproduction and limits", "",
        "Run `python -m capalloc.reproduce` after unpacking the companion source snapshot. The frozen transformer uses at most four 256-token chunks from selected excerpts, with masked mean pooling. `reports/transformer_embeddings.audit.json` gives actual truncation counts. The chosen passages and pretrained encoder may carry firm or time signals beyond the intended financial language. The 24-firm text subset is selective; company clustering does not account for shared year shocks. No causal or trading claim follows from these results.", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=Path, default=Path("reports/metrics.json"))
    p.add_argument("--predictions", type=Path, default=Path("reports/test_predictions.csv"))
    p.add_argument("--panel", type=Path, default=Path("data/processed/panel.csv"))
    p.add_argument("--unseen", type=Path, default=Path("reports/unseen_company.json"))
    p.add_argument("--secondary", type=Path, default=Path("reports/secondary.json"))
    p.add_argument("--out", type=Path, default=Path("docs/RESULTS.md"))
    a = p.parse_args()
    make_report(a.metrics, a.predictions, a.panel, a.unseen, a.secondary, a.out)


if __name__ == "__main__":
    main()

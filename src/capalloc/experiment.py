"""Availability-aware temporal ablations and paired exploratory inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

NUMERIC = ["capex_rate", "repurchase_rate", "cfo_rate", "rnd_rate", "dividend_rate",
           "revenue_growth", "capex_delta", "repurchase_delta", "log_revenue", "year"]
SPLIT = {"training_labels_before": "2022-01-01", "validation_predictions_from": "2022-01-01",
         "validation_predictions_before": "2023-01-01", "validation_labels_before": "2024-01-01",
         "test_predictions_from": "2024-01-01", "test_predictions_before": "2026-01-01"}


def load_panel(path):
    d = pd.read_csv(path, dtype={"cik": str}).sort_values(["cik", "fiscal_end"]).copy()
    if not d.id.is_unique or not (d.known_at < d.target_known_at).all():
        raise ValueError("Duplicate IDs or target published before its source filing")
    gap = (pd.to_datetime(d.fiscal_end) - pd.to_datetime(d.groupby("cik").fiscal_end.shift(1))).dt.days
    earlier = d.groupby("cik").known_at.shift(1) < d.known_at
    for name, source in [("revenue_growth", "revenue"), ("capex_delta", "capex_rate"), ("repurchase_delta", "repurchase_rate")]:
        lag = d.groupby("cik")[source].shift(1).where(gap.between(330, 400) & earlier)
        d[name] = d[source] / lag - 1 if name == "revenue_growth" else d[source] - lag
    d["log_revenue"] = np.log(d["revenue"].clip(lower=1))
    d["revenue_growth"] = d["revenue_growth"].replace([np.inf, -np.inf], np.nan)
    return d.sort_values(["known_at", "cik"])


def split(d):
    tr = d[d.target_known_at < SPLIT["training_labels_before"]].copy()
    va = d[(d.known_at >= SPLIT["validation_predictions_from"]) &
           (d.known_at < SPLIT["validation_predictions_before"]) &
           (d.target_known_at < SPLIT["validation_labels_before"])].copy()
    te = d[(d.known_at >= SPLIT["test_predictions_from"]) &
           (d.known_at < SPLIT["test_predictions_before"])].copy()
    if len(tr) and len(va) and tr.target_known_at.max() >= va.known_at.min():
        raise ValueError("Unpublished training outcome at validation time")
    if len(va) and len(te) and va.target_known_at.max() >= te.known_at.min():
        raise ValueError("Unpublished validation outcome at test time")
    return tr, va, te


def numeric_preprocessor():
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler()), NUMERIC),
        ("category", OneHotEncoder(handle_unknown="ignore"), ["sector", "capex_tag"]),
    ], sparse_threshold=0)


def matrix(pre, tr, va, te):
    a = pre.fit_transform(tr); b = pre.transform(va); c = pre.transform(te)
    return a, b, c


def evaluate(y, pred):
    return {"n": len(y), "mae_pp": float(mean_absolute_error(y, pred) * 100),
            "rmse_pp": float(np.sqrt(mean_squared_error(y, pred)) * 100)}


def paired_cluster_test(ids, company, y, base, candidate, seed=41, repeats=20000):
    """Company cluster bootstrap CI and sign-flip randomization on paired loss deltas.

    Improvement >0 means the candidate has lower absolute error. No iid row assumption.
    """
    ids, company, y, base, candidate = [np.asarray(v) for v in (ids, company, y, base, candidate)]
    if len({len(v) for v in (ids, company, y, base, candidate)}) != 1 or not len(y):
        raise ValueError("Paired inputs must be nonempty and equal length")
    if len(np.unique(ids)) != len(ids) or not np.isfinite(np.concatenate((y, base, candidate))).all():
        raise ValueError("Paired inputs require unique IDs and finite values")
    delta = np.abs(y - base) - np.abs(y - candidate)
    unique = np.unique(company)
    groups = [np.where(company == c)[0] for c in unique]
    rng = np.random.default_rng(seed)
    means = np.empty(repeats)
    flips = np.empty(repeats)
    for i in range(repeats):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        means[i] = delta[idx].mean()
        signs = rng.choice((-1, 1), len(groups))
        flips[i] = sum(signs[j] * delta[g].sum() for j, g in enumerate(groups)) / len(delta)
    observed = float(delta.mean())
    return {"comparison": "baseline minus candidate absolute error; positive favors candidate",
            "n_rows": len(ids), "n_companies": len(groups), "improvement_pp": observed * 100,
            "cluster_bootstrap_95_ci_pp": [float(x * 100) for x in np.quantile(means, [.025, .975])],
            "two_sided_cluster_sign_flip_p": float((1 + np.sum(np.abs(flips) >= abs(observed))) / (repeats + 1)),
            "seed": seed, "replicates": repeats}


def validation_interval(va_y, va_pred, test_pred, alpha=.1):
    residuals = np.sort(np.abs(va_y - va_pred))
    k = math.ceil((len(residuals) + 1) * (1 - alpha))
    q = float(residuals[min(k, len(residuals)) - 1]) if k <= len(residuals) else float("inf")
    return test_pred - q, test_pred + q, q


def fit_ridge(a, y, b, yb, c):
    scores = []
    for alpha in [0.1, 1., 10., 100., 1000., 10000.]:
        m = Ridge(alpha=alpha).fit(a, y)
        scores.append((mean_absolute_error(yb, m.predict(b)), alpha))
    _, best = min(scores)
    m = Ridge(alpha=best).fit(a, y)
    return m.predict(b), m.predict(c), {"alpha": best}


def fit_residual_ridge(a, y, b, yb, c, anchor_a, anchor_b, anchor_c):
    vv, pp, choice = fit_ridge(a, y - anchor_a, b, yb - anchor_b, c)
    return anchor_b + vv, anchor_c + pp, choice


def embed_transformer(texts, cache_path, model_name, revision, max_length=256, batch_size=8):
    """Frozen encoder with up to four chunks per document and a content-keyed cache."""
    import hashlib
    import torch
    from transformers import AutoModel, AutoTokenizer
    digest = hashlib.sha256(json.dumps({"texts": list(texts), "model": model_name, "revision": revision,
                                       "max_length": max_length, "max_chunks": 4, "pool": 2}, sort_keys=True).encode()).hexdigest()
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=False)
        if str(payload["digest"]) == digest:
            return payload["embeddings"]
    torch.set_num_threads(2)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, revision=revision, local_files_only=True).eval()
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModel.from_pretrained(model_name, revision=revision).eval()
    chunks, owners, totals, used = [], [], [], []
    for owner, passage in enumerate(texts):
        tokens = tokenizer.encode(passage, add_special_tokens=False, verbose=False)
        totals.append(len(tokens))
        selected = tokens[:4 * (max_length - 2)]
        used.append(len(selected))
        for start in range(0, max(1, len(selected)), max_length - 2):
            chunks.append(tokenizer.prepare_for_model(selected[start:start + max_length - 2], add_special_tokens=True))
            owners.append(owner)
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(chunks), batch_size):
            batch = tokenizer.pad(chunks[start:start + batch_size], padding=True, return_tensors="pt")
            out = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pool = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            vectors.append(pool.numpy())
    chunk_vectors, owners = np.concatenate(vectors), np.asarray(owners)
    embeddings = np.stack([chunk_vectors[owners == i].mean(0) for i in range(len(texts))]).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.with_suffix(".audit.json").write_text(json.dumps({"documents": len(texts),
        "chunks": len(owners), "max_chunks": 4, "truncated_documents": sum(a > b for a, b in zip(totals, used)),
        "total_tokens": totals, "used_tokens": used}, indent=2))
    np.savez_compressed(cache_path, embeddings=embeddings, digest=np.array(digest))
    return embeddings


def run(panel, out_dir, transformer=False, gru=False, model_name="ProsusAI/finbert", revision="0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"):
    out_dir.mkdir(parents=True, exist_ok=True)
    d = load_panel(panel)
    report = {"protocol": {"target": "next fiscal year capex / next fiscal year revenue",
                           "split": SPLIT, "study_status": "retrospective exploratory audit revision; earlier test results were inspected",
                           "primary_hypothesis": "adding frozen filing embeddings reduces held-out MAE relative to the same numeric residual ridge without text",
                           "transformer_executed": transformer, "gru_executed": gru,
                           "source_panel_sha256": hashlib.sha256(panel.read_bytes()).hexdigest()},
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "pandas": pd.__version__, "scikit_learn": sklearn.__version__}, "cohorts": {}}
    if transformer or gru:
        import torch
        report["runtime"]["torch"] = torch.__version__
    if transformer:
        import transformers
        report["runtime"]["transformers"] = transformers.__version__
    all_predictions = []
    for cohort in ("all", "text"):
        subset = d if cohort == "all" else d[d.has_text == 1]
        tr, va, te = split(subset)
        if min(len(tr), len(va), len(te)) < 8:
            raise ValueError(f"Too few {cohort} rows for chronological evaluation: {[len(x) for x in (tr, va, te)]}")
        y, yv, yt = [x.target_capex_rate.to_numpy() for x in (tr, va, te)]
        xa, xb, xc = matrix(numeric_preprocessor(), tr, va, te)
        results, predictions = {}, {}
        persistence_v, persistence_t = va.capex_rate.to_numpy(), te.capex_rate.to_numpy()
        results["persistence"] = evaluate(yt, persistence_t); predictions["persistence"] = (persistence_v, persistence_t)
        peers = tr.groupby("sector").target_capex_rate.median()
        global_median = tr.target_capex_rate.median()
        peer_v = va.sector.map(peers).fillna(global_median).to_numpy()
        peer_t = te.sector.map(peers).fillna(global_median).to_numpy()
        results["sector_median"] = evaluate(yt, peer_t); predictions["sector_median"] = (peer_v, peer_t)
        pv, pt, chosen = fit_ridge(xa, y, xb, yv, xc)
        results["numeric_ridge"] = {**evaluate(yt, pt), "validation_choice": chosen}; predictions["numeric_ridge"] = (pv, pt)
        rv, rt, choice = fit_residual_ridge(xa, y, xb, yv, xc, tr.capex_rate.to_numpy(), persistence_v, persistence_t)
        results["numeric_residual_ridge"] = {**evaluate(yt, rt), "validation_choice": choice}
        predictions["numeric_residual_ridge"] = (rv, rt)
        candidates = []
        for depth in [2, 4, None]:
            m = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=12, max_depth=depth,
                                              l2_regularization=1., min_samples_leaf=15, random_state=41).fit(xa, y)
            candidates.append((mean_absolute_error(yv, m.predict(xb)), depth, m))
        _, depth, m = min(candidates, key=lambda x: x[0])
        tv, tt = m.predict(xb), m.predict(xc)
        results["numeric_tree"] = {**evaluate(yt, tt), "validation_choice": {"max_depth": depth}}
        predictions["numeric_tree"] = (tv, tt)
        residual_candidates = []
        for depth in [2, 4, None]:
            r = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=12, max_depth=depth,
                                              l2_regularization=1., min_samples_leaf=15, random_state=41).fit(
                                                  xa, y - tr.capex_rate.to_numpy())
            residual_candidates.append((mean_absolute_error(yv, persistence_v + r.predict(xb)), depth, r))
        _, depth, r = min(residual_candidates, key=lambda x: x[0])
        vvr, ttr = persistence_v + r.predict(xb), persistence_t + r.predict(xc)
        results["numeric_residual_tree"] = {**evaluate(yt, ttr), "validation_choice": {"max_depth": depth}}
        predictions["numeric_residual_tree"] = (vvr, ttr)
        numeric_names = ["persistence", "sector_median", "numeric_ridge", "numeric_residual_ridge", "numeric_tree", "numeric_residual_tree"]
        if gru:
            from .sequence import train_gru
            gv, gt, settings = train_gru(tr, va, te, history_panel=d)
            results["numeric_gru"] = {**evaluate(yt, gt), "validation_choice": settings}
            predictions["numeric_gru"] = (gv, gt)
            numeric_names.append("numeric_gru")
        best_numeric = min(numeric_names, key=lambda name: mean_absolute_error(yv, predictions[name][0]))
        report["cohorts"].setdefault(cohort, {})["validation_selected_numeric"] = best_numeric
        if cohort == "text":
            vec = TfidfVectorizer(min_df=2, max_df=.95, max_features=5000, ngram_range=(1, 2),
                                  stop_words="english", sublinear_tf=True)
            ta = vec.fit_transform(tr.text.fillna("")); tb = vec.transform(va.text.fillna("")); tc = vec.transform(te.text.fillna(""))
            for variant, a, b, c in [
                ("text_tfidf", ta, tb, tc),
                ("numeric_plus_tfidf", sparse.hstack([sparse.csr_matrix(xa), ta]).tocsr(),
                 sparse.hstack([sparse.csr_matrix(xb), tb]).tocsr(), sparse.hstack([sparse.csr_matrix(xc), tc]).tocsr()),
            ]:
                vv, pp, choice = fit_ridge(a, y, b, yv, c)
                results[variant] = {**evaluate(yt, pp), "validation_choice": choice}; predictions[variant] = (vv, pp)
            if transformer:
                combined = pd.concat([tr, va, te])
                embeddings = embed_transformer(combined.text.fillna("").tolist(), out_dir / "transformer_embeddings.npz", model_name, revision)
                ea, eb, ec = np.split(embeddings, [len(tr), len(tr) + len(va)])
                scaler = StandardScaler().fit(ea)
                ea, eb, ec = [scaler.transform(x) for x in (ea, eb, ec)]
                for variant, a, b, c in [
                    ("transformer_text", ea, eb, ec),
                    ("numeric_plus_transformer", np.column_stack([xa, ea]), np.column_stack([xb, eb]), np.column_stack([xc, ec])),
                ]:
                    if variant == "numeric_plus_transformer":
                        vv, pp, choice = fit_residual_ridge(a, y, b, yv, c, tr.capex_rate.to_numpy(), persistence_v, persistence_t)
                    else:
                        vv, pp, choice = fit_ridge(a, y, b, yv, c)
                    results[variant] = {**evaluate(yt, pp), "validation_choice": choice}; predictions[variant] = (vv, pp)
                report["transformer"] = {"model": model_name, "revision": revision,
                                          "pooling": "masked token mean then equal chunk mean",
                                          "tokens_per_chunk": 256, "max_chunks": 4, "fine_tuned": False}
                report["hypothesis_test"] = paired_cluster_test(te.id.to_numpy(), te.cik.to_numpy(), yt,
                       predictions["numeric_residual_ridge"][1], predictions["numeric_plus_transformer"][1])
                report["hypothesis_test"]["baseline"] = "numeric_residual_ridge"
                report["benchmark_comparison"] = paired_cluster_test(te.id.to_numpy(), te.cik.to_numpy(), yt,
                       predictions[best_numeric][1], predictions["numeric_plus_transformer"][1])
                report["benchmark_comparison"]["baseline"] = best_numeric
        lo, hi, q = validation_interval(yv, pv, pt)
        results["numeric_ridge"]["interval_90"] = {"validation_quantile_pp": q * 100,
            "coverage": float(np.mean((yt >= lo) & (yt <= hi))), "mean_width_pp": float(np.mean(hi - lo) * 100),
            "note": "validation residual diagnostic; validation also chose hyperparameters, so no formal coverage guarantee"}
        report["cohorts"][cohort].update({"counts": {"train": len(tr), "validation": len(va), "test": len(te),
                                                  "test_companies": int(te.cik.nunique())}, "models": results,
            "availability_audit": {"latest_training_label": tr.target_known_at.max(),
              "earliest_validation_prediction": va.known_at.min(),
              "latest_validation_label": va.target_known_at.max(),
              "earliest_test_prediction": te.known_at.min()},
            "validation_mae_pp": {name: float(mean_absolute_error(yv, pair[0]) * 100)
                                  for name, pair in predictions.items()}})
        for j, row in enumerate(te.itertuples(index=False)):
            result = {"cohort": cohort, "id": row.id, "cik": row.cik, "ticker": row.ticker,
                      "sector": row.sector, "fiscal_end": row.fiscal_end, "known_at": row.known_at,
                      "target_known_at": row.target_known_at, "source_accession": row.source_accession,
                      "target_accession": row.target_accession, "actual": float(yt[j]),
                      "numeric_interval_low": float(lo[j]), "numeric_interval_high": float(hi[j])}
            result.update({k: float(v[1][j]) for k, v in predictions.items()})
            all_predictions.append(result)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame(all_predictions).to_csv(out_dir / "test_predictions.csv", index=False)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, default=Path("data/processed/panel.csv"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--transformer", action="store_true")
    p.add_argument("--gru", action="store_true")
    p.add_argument("--model", default="ProsusAI/finbert")
    p.add_argument("--revision", default="0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031")
    args = p.parse_args()
    print(json.dumps(run(args.panel, args.out, args.transformer, args.gru, args.model, args.revision), indent=2))


if __name__ == "__main__":
    main()

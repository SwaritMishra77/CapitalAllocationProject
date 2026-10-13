# Results

## Primary paired comparison

On **32 test filings from 23 companies with cached text**, the numeric residual ridge had 0.454 percentage points (pp) mean absolute error (MAE). The same numeric model with frozen FinBERT embeddings had 0.433 pp MAE. Mean paired improvement was **+0.021 pp** (positive favors text); company-cluster bootstrap 95% interval **[-0.004, +0.044] pp**; two-sided company sign-flip p = **0.123**. The interval includes zero. This sample does not establish a reliable benefit from the transformer.

Against the validation-selected numeric benchmark (persistence), the same paired comparison was +0.030 pp (95% cluster interval [-0.015, +0.075] pp). This is a distinct benchmark, selected on validation rather than for the fixed ablation. Both are exploratory: earlier test results were inspected before this availability audit and four-chunk revision. No confirmatory p-value is claimed.

The 100-company source universe produced 751 training, 93 validation and 134 held-out numerical rows (92 test firms). Cached filing text is available for only 24 source firms; the matched text experiment has 86/23/32 rows and 23 test firms. These are not random text coverage or financial returns.

## Publication-date gate

Forecast features are available at the source 10-K filing date. Targets enter training or tuning only after the *next* original 10-K was filed. These gates prevent later outcomes from entering an earlier decision:

| Cohort | Latest train label | First validation forecast | Latest validation label | First test forecast |
| --- | --- | --- | --- | --- |
| All | 2021-12-21 | 2022-01-25 | 2023-12-20 | 2024-01-23 |
| Text | 2021-12-16 | 2022-01-27 | 2023-12-15 | 2024-01-25 |

The prior fiscal-year split in the first release allowed labels that would not yet have been published. Its scores and sample sizes have been superseded by this revision. See [audit notes](AUDIT.md).

## Held-out forecast error

MAE is measured in percentage points of next-year capital expenditure divided by next-year revenue. All choices and preprocessing use earlier available data.

| Model | Full cohort MAE (pp) | Text-matched MAE (pp) |
| --- | ---: | ---: |
| persistence | 0.486 | 0.463 |
| sector median | 1.099 | 1.212 |
| numeric ridge | 0.578 | 0.670 |
| numeric residual ridge | 0.472 | 0.454 |
| numeric tree | 0.597 | 0.831 |
| numeric residual tree | 0.574 | 0.623 |
| numeric gru | 0.828 | 0.945 |
| text tfidf | — | 0.818 |
| numeric plus tfidf | — | 0.669 |
| transformer text | — | 1.038 |
| numeric plus transformer | — | 0.433 |

The numeric ridge 90% **validation residual diagnostic** covered 91.8% of test rows with mean width 2.59 pp. Validation residuals also chose model settings, and time distributions can shift; this interval has no formal 90% coverage guarantee.

## Matched subgroup descriptions

Every subgroup compares numeric_residual_ridge and numeric-plus-transformer on the same rows. Small groups are descriptive only.

| Segment | Rows | Numbers MAE (pp) | Numbers + text MAE (pp) |
| --- | ---: | ---: | ---: |
| manufacturing | 20 | 0.444 | 0.407 |
| retail | 12 | 0.470 | 0.475 |
| 2023 | 10 | 0.409 | 0.389 |
| 2024 | 22 | 0.474 | 0.452 |
| At/above train median revenue | 25 | 0.425 | 0.397 |
| Below train median revenue | 7 | 0.557 | 0.558 |

## Largest numerical errors

These observations warrant filing review; model error is not an analyst alert.

| Company | Source fiscal year | Actual (pp) | Forecast (pp) | Error (pp) |
| --- | ---: | ---: | ---: | ---: |
| BURL | 2024 | 8.28 | 4.68 | 3.60 |
| FND | 2024 | 6.78 | 9.26 | 2.48 |
| DKS | 2024 | 5.97 | 4.15 | 1.82 |
| VMI | 2024 | 3.53 | 1.81 | 1.72 |
| AZO | 2024 | 7.01 | 5.30 | 1.71 |

## Unseen-company check

A fixed hash-based split kept 20 companies out of model fitting. Its 29 test rows (20 firms) gave persistence MAE 0.452 pp and numerical residual ridge MAE 0.559 pp. This is a numerical portability check only.

## Other reported spending

Secondary targets require both source and next-year tags to be observed. A missing tag is never assumed to be zero. R&D is an expense and is analyzed separately from capital cash spending.

| Next-year ratio | Test rows | Persistence MAE (pp) | Numerical residual ridge MAE (pp) |
| --- | ---: | ---: | ---: |
| repurchase / revenue | 117 | 2.868 | 3.380 |
| dividend / revenue | 104 | 0.583 | 0.574 |
| rnd / revenue | 66 | 0.271 | 0.276 |

## Reproduction and limits

Run `python -m capalloc.reproduce` after unpacking the companion source snapshot. The frozen transformer uses at most four 256-token chunks from selected excerpts, with masked mean pooling. `reports/transformer_embeddings.audit.json` gives actual truncation counts. The chosen passages and pretrained encoder may carry firm or time signals beyond the intended financial language. The 24-firm text subset is selective; company clustering does not account for shared year shocks. No causal or trading claim follows from these results.

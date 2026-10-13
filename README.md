# Capital Allocation Follow-Through

Can the language in a company's annual report improve a forecast of what it actually spends on capital projects the following year? This repository builds a point-in-time panel of original SEC 10-K filings and predicts **next fiscal year's cash capital expenditure divided by next fiscal year's revenue**. It compares persistence, numerical machine learning, a small sequence model, filing text and a frozen financial text encoder on held-out filings.

**Start with the [results](docs/RESULTS.md), [interactive filing explorer](docs/index.html), and [audit trail](docs/AUDIT.md).** The key test directly compares numeric residual ridge with the same numeric model plus FinBERT embeddings on identical text-covered observations. A more complex model earns no credit for sounding sophisticated: the paired interval includes zero. This is a retrospective research study, not a trading system or a claim about management intent.

## What is in the repository

- `src/capalloc/` contains **all Python source** for collection, panel construction, source reconciliation, evaluation, models and the explorer.
- `tests/` tests date availability, annual lag construction, source selection and paired evaluation. `.github/workflows/ci.yml` runs them on pushes.
- `reports/` contains actual metrics, source quality checks, reconciliation records and every held-out prediction. `docs/index.html` is a standalone page suitable for GitHub Pages.
- `configs/` pins the 100-company source universe and a manifest of 408 source files. Only 99 companies yield eligible paired years; CNH does not. The matched text test is much smaller because original HTML was cached for 24 firms.
- `data/raw/` is deliberately empty in Git. The separate `CapitalAllocation_SourceSnapshot.zip` contains the original SEC responses and filing HTML needed for exact reproduction. Model weights are downloaded from their public source.

## Reproduce the published experiment

Use **Python 3.12**. From the repository root, create an environment and install the CPU version of PyTorch, the pinned research packages and this package:

```bash
python -m venv .venv
# On macOS/Linux: source .venv/bin/activate
# On Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-reproduce.txt
python -m pip install -e .
```

Extract the **contents** of `CapitalAllocation_SourceSnapshot.zip` directly under `data/raw/`. The expected structure is `data/raw/facts/`, `data/raw/filings/`, `data/raw/html/`, and `data/raw/universe_resolved.json`. Then:

```bash
python -m capalloc.reproduce
```

The single command verifies the snapshot against the manifest, runs tests, builds and audits the panel, reconciles selected source facts against inline XBRL in original filings, runs the numerical, GRU and transformer experiments, computes robustness/secondary results, and regenerates `docs/RESULTS.md` and `docs/index.html`. The initial model download is substantial; CPU inference may take several minutes. Subsequent runs cache a content-keyed embedding matrix locally. The transformer uses public `ProsusAI/finbert` pinned to commit `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`.

Without the optional model weights, `python -m pytest -q` checks the code, and `python -m capalloc.verify --raw data/raw` verifies a supplied snapshot. No SEC key is required for a fresh data pull, but a real contact `SEC_USER_AGENT` is required; see `python -m capalloc.ingest --help`. A new pull is **a different snapshot** and requires a fresh audit and new result claims.

## Design in brief

The source filing date determines when a forecast can be made. Training outcomes must be published before 2022; validation forecasts use filings in 2022 with targets published before 2024; the test uses filings from 2024–2025. Earlier source-year lags require an adjacent fiscal year and a prior available filing. Each annual USD fact is selected from the filing's **own original accession** for a 330–400-day period, so later comparative restatements do not overwrite inputs. Missing reported tags stay missing. `docs/METHOD.md` describes the decisions and limitations in detail.

The original release used a fiscal-year split that admitted unpublished labels. This version corrects the boundary, reruns the outputs and explicitly treats its hypothesis test as exploratory because earlier test results were inspected. The change and numeric checks are documented in [AUDIT.md](docs/AUDIT.md).

## Publish and inspect

Commit this project tree to GitHub, including `src/`, `tests/`, `docs/`, `reports/`, `configs/`, workflow and package files. Leave the large raw snapshot, generated panel, embeddings and virtual environment out of Git. For a public explorer link, configure **Settings → Pages → Deploy from a branch → main /docs**. The resulting `https://<username>.github.io/<repository>/` URL works for others; a local `file:///` path does not.

Source: [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces). Frozen language encoder: [ProsusAI FinBERT](https://huggingface.co/ProsusAI/finbert). Code: MIT license. Source filings and model weights retain their respective terms.

"""Build the source-only GitHub ZIP, never bundling raw SEC data or model cache."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = (".github", "configs", "docs", "reports", "scripts", "src", "tests")
FILES = (".gitattributes", ".gitignore", "LICENSE", "README.md", "pyproject.toml", "requirements-reproduce.txt")
SKIP_PARTS = {".git", ".venv", ".pytest_cache", "__pycache__"}
SKIP_SUFFIXES = (".pyc", ".pyo", ".npz", ".tmp", ".log")


def build(path):
    selected = [ROOT / name for name in FILES]
    for name in DIRECTORIES:
        selected.extend(p for p in (ROOT / name).rglob("*") if p.is_file()
                        and not any(part in SKIP_PARTS or part.endswith(".egg-info") for part in p.parts)
                        and not p.name.endswith(SKIP_SUFFIXES))
    selected.extend((ROOT / "data" / branch / ".gitkeep") for branch in ("raw", "processed"))
    missing = [str(p) for p in selected if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for file in sorted(set(selected)):
            archive.write(file, "CapitalAllocation-FollowThrough/" + file.relative_to(ROOT).as_posix())
    return len(set(selected))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT.parent / "CapitalAllocation_FollowThrough_GitHub.zip")
    args = parser.parse_args()
    print(f"Packaged {build(args.out)} files in {args.out}")

"""Cross-platform single command to reconstruct all published outputs."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    if not (args.raw / "universe_resolved.json").exists():
        parser.error("Source snapshot missing. Unpack companion ZIP into data/raw; see README.md")
    commands = [
        [sys.executable, "-m", "capalloc.verify", "--raw", str(args.raw)],
        [sys.executable, "-m", "pytest", "-q"],
        [sys.executable, "-m", "capalloc.panel", "--raw", str(args.raw), "--out", "data/processed", "--max-year", "2025"],
        [sys.executable, "-m", "capalloc.quality"],
        [sys.executable, "-m", "capalloc.reconcile", "--raw", str(args.raw)],
        [sys.executable, "-m", "capalloc.experiment", "--gru", "--transformer"],
        [sys.executable, "-m", "capalloc.robustness"],
        [sys.executable, "-m", "capalloc.secondary"],
        [sys.executable, "-m", "capalloc.report"],
        [sys.executable, "-m", "capalloc.demo"],
    ]
    for command in commands:
        print("+ " + " ".join(command), flush=True)
        subprocess.run(command, check=True, env=os.environ.copy())
    print("Finished: docs/index.html and docs/RESULTS.md; metrics in reports/.", flush=True)


if __name__ == "__main__":
    main()

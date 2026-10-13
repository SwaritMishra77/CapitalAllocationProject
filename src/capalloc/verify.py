"""Check every cached SEC byte against the pinned source manifest."""

import argparse
import hashlib
import json
from pathlib import Path


def verify(raw, manifest):
    expected = json.loads(manifest.read_text())["files"]
    bad = []
    for name, meta in expected.items():
        p = raw / name
        if not p.exists() or p.stat().st_size != meta["bytes"] or hashlib.sha256(p.read_bytes()).hexdigest() != meta["sha256"]:
            bad.append(name)
    if bad:
        raise ValueError(f"Source snapshot failed verification for {len(bad)} files: {bad[:5]}")
    return len(expected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--manifest", type=Path, default=Path("configs/snapshot_manifest.json"))
    a = p.parse_args()
    print(f"Verified {verify(a.raw, a.manifest)} source files")


if __name__ == "__main__":
    main()

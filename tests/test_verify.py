import hashlib
import json

import pytest

from capalloc.verify import verify


def test_manifest_detects_changed_source_bytes(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "source.json"
    source.write_bytes(b"original")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": {"source.json": {"bytes": 8,
        "sha256": hashlib.sha256(b"original").hexdigest()}}}))
    assert verify(raw, manifest) == 1
    source.write_bytes(b"changed!")
    with pytest.raises(ValueError, match="failed verification"):
        verify(raw, manifest)

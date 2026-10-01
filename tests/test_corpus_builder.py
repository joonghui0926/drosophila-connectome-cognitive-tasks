import importlib.util
import json
from pathlib import Path
import sys


def test_builder_manifest_and_snapshot_work_outside_repository(tmp_path, monkeypatch):
    script = Path(__file__).parents[1] / "scripts" / "build_language_corpus.py"
    spec = importlib.util.spec_from_file_location("build_language_corpus", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def write_fixture(path, **kwargs):
        path.write_bytes(b"fixture")
        return {"path": path.name, "sha256": module.sha256(path)}

    monkeypatch.setattr(module, "write_split", write_fixture)
    monkeypatch.setattr(module, "write_unique_bank", lambda source, destination: write_fixture(destination))
    monkeypatch.setattr(module, "write_samples", lambda path, **kwargs: path.write_text("samples"))
    monkeypatch.setattr(module, "audit_corpus", lambda **kwargs: {"fixture": True})
    monkeypatch.setattr(module, "audit_files", lambda path: {"fixture": True})
    output = tmp_path / "corpus"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(script), "--output-dir", str(output)])
    module.main()

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["builder"]["path"] == "scripts/build_language_corpus.py"
    assert manifest["builder"]["sha256"] == module.sha256(script)
    assert manifest["generator"]["path"] == "src/flystudy/language_corpus.py"
    assert len(manifest["files"]) == 5
    assert len(manifest["sourceSnapshot"]) == 6
    for name, expected in manifest["sourceSnapshot"].items():
        assert module.sha256(output / "source-snapshot" / name) == expected

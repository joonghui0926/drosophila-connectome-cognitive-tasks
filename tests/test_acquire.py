import hashlib
import io
import json

import pytest

import flystudy.acquire as acquire


def source_for(content):
    return {
        "name": "fixture.feather",
        "url": "https://example.invalid/fixture.feather",
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def test_download_is_verified_before_publication(tmp_path, monkeypatch):
    content = b"verified data"
    monkeypatch.setattr(acquire, "FILES", {"fixture": source_for(content)})
    monkeypatch.setattr(acquire.urllib.request, "urlopen", lambda request: io.BytesIO(content))
    result = acquire.acquire_malecns(tmp_path)
    assert (tmp_path / "fixture.feather").read_bytes() == content
    assert not (tmp_path / "fixture.feather.part").exists()
    assert result["files"]["fixture"]["verified"]
    assert json.loads((tmp_path / "sources.json").read_text()) == result


@pytest.mark.parametrize("existing", [None, b"old cached data"])
def test_invalid_download_does_not_replace_the_destination(tmp_path, monkeypatch, existing):
    content = b"verified data"
    target = tmp_path / "fixture.feather"
    if existing is not None:
        target.write_bytes(existing)
    monkeypatch.setattr(acquire, "FILES", {"fixture": source_for(content)})
    monkeypatch.setattr(acquire.urllib.request, "urlopen", lambda request: io.BytesIO(b"x" * len(content)))
    with pytest.raises(RuntimeError, match="verification failed"):
        acquire.acquire_malecns(tmp_path)
    if existing is None:
        assert not target.exists()
    else:
        assert target.read_bytes() == existing
    assert not (tmp_path / "sources.json").exists()


def test_corrupt_same_size_cache_is_downloaded_again(tmp_path, monkeypatch):
    content = b"verified data"
    (tmp_path / "fixture.feather").write_bytes(b"x" * len(content))
    monkeypatch.setattr(acquire, "FILES", {"fixture": source_for(content)})
    monkeypatch.setattr(acquire.urllib.request, "urlopen", lambda request: io.BytesIO(content))
    acquire.acquire_malecns(tmp_path)
    assert (tmp_path / "fixture.feather").read_bytes() == content


def test_valid_cache_does_not_access_the_network(tmp_path, monkeypatch):
    content = b"verified data"
    (tmp_path / "fixture.feather").write_bytes(content)
    monkeypatch.setattr(acquire, "FILES", {"fixture": source_for(content)})

    def unexpected_download(request):
        pytest.fail("verified cache should not be downloaded again")

    monkeypatch.setattr(acquire.urllib.request, "urlopen", unexpected_download)
    acquire.acquire_malecns(tmp_path)

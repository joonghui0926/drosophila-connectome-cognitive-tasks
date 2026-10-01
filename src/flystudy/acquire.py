from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from flystudy import __version__


FILES = {
    "annotations": {
        "name": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "bytes": 14_483_314,
        "sha256": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    },
    "connectivity": {
        "name": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "bytes": 1_051_241_946,
        "sha256": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_malecns(destination: str | Path) -> dict:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    for key, source in FILES.items():
        target = destination / source["name"]
        verified = (target.is_file() and target.stat().st_size == source["bytes"]
                    and _sha256(target) == source["sha256"])
        if not verified:
            partial = target.with_suffix(target.suffix + ".part")
            request = urllib.request.Request(
                source["url"], headers={"User-Agent": f"flystudy/{__version__}"}
            )
            with urllib.request.urlopen(request) as response, partial.open("wb") as output:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
            if partial.stat().st_size != source["bytes"] or _sha256(partial) != source["sha256"]:
                raise RuntimeError(f"MaleCNS file verification failed: {partial}")
            partial.replace(target)
        records[key] = {**source, "path": str(target), "verified": True}
    manifest = {
        "dataset": "MaleCNS v1.0",
        "license": "CC BY 4.0",
        "download_page": "https://male-cns.janelia.org/download/",
        "files": records,
    }
    (destination / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

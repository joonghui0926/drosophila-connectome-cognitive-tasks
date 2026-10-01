"""Model construction, interface fingerprints and result-file utilities."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from flystudy.language_anatomy import anatomical_sensory_groups
from flystudy.language_model import ConnectomeLanguageModel


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    os.replace(temporary, path)


def build_model(args, device: torch.device):
    graph_dir = Path(args.graph_dir)
    manifest = json.loads((graph_dir / "manifest.json").read_text())
    graph = np.load(graph_dir / "graph.npz")
    sensory_groups, sensory_metadata = anatomical_sensory_groups(graph_dir)
    options = {
        "seed": args.seed,
        "interface_seed": args.interface_seed,
        "embedding": args.embedding,
        "dimension": args.dimension,
        "token_group": args.token_group,
        "output_group": args.output_group,
        "token_steps": args.token_steps,
        "leak": args.leak,
        "token_gain": args.token_gain,
        "sensor_gain": args.sensor_gain,
        "temperature": args.temperature,
        "backend": args.backend,
        "chunk_size": args.chunk_size,
        "token_drive_mode": getattr(args, "token_drive_mode", "sustained"),
    }
    model = ConnectomeLanguageModel(
        graph["rows"], graph["cols"], manifest["neurons"],
        sensory_groups=sensory_groups, **options).to(device)
    parameters = dict(model.named_parameters())
    if list(parameters) != ["theta"] or model.theta.numel() != manifest["edges"]:
        raise RuntimeError("connectome parameter audit failed")
    return model, manifest, sensory_metadata, options


def interface_fingerprint(model: ConnectomeLanguageModel) -> str:
    digest = hashlib.sha256()
    for name in ("sensor_projection", "token_nodes", "output_nodes",
                 "input_codebook", "output_codebook"):
        digest.update(name.encode())
        digest.update(getattr(model, name).detach().cpu().numpy().tobytes())
    return digest.hexdigest()

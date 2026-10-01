from __future__ import annotations

import base64
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch


def stable_spatial_indices(
    nodes_path: str | Path,
    *,
    seed: int = 20260912,
    spatial_points: int = 3_000,
) -> np.ndarray:
    nodes = feather.read_table(nodes_path, columns=["somaLocation"], memory_map=True)
    locations = nodes["somaLocation"].to_pylist()
    valid = np.asarray([index for index, value in enumerate(locations) if value is not None], dtype=np.int64)
    count = min(int(spatial_points), len(valid))
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(valid, size=count, replace=False))


def spatial_metadata(nodes_path: str | Path, indices: np.ndarray) -> list[dict]:
    nodes = feather.read_table(
        nodes_path,
        columns=["bodyId", "superclass", "somaLocation"],
        memory_map=True,
    )
    body_ids = nodes["bodyId"].to_numpy(zero_copy_only=False)
    superclasses = np.asarray(nodes["superclass"].to_pylist(), dtype=object)
    locations = nodes["somaLocation"].to_pylist()
    chosen = np.asarray(indices, dtype=np.int64)
    result = []
    for index in chosen:
        location = locations[int(index)]
        if location is None:
            raise ValueError("spatial metadata requires soma-bearing node indices")
        x, y, z = location
        result.append(
            {
                "bodyId": int(body_ids[index]),
                "x": int(x),
                "y": int(y),
                "z": int(z),
                "activity": 0.0,
                "normalized": 0.0,
                "superclass": str(superclasses[index]),
            }
        )
    return result


def encode_activity_replay(selected_activity: torch.Tensor) -> dict:
    """Quantize [step, point, probe] magnitudes into compact aligned frames."""
    if selected_activity.ndim != 3:
        raise ValueError("selected activity must have shape [steps, points, probes]")
    values = selected_activity.detach().abs().float().cpu().numpy()
    scale = float(np.quantile(values, 0.99)) if values.size else 1.0
    normalized = np.clip(values / max(scale, 1e-8), 0.0, 1.0)
    encoded = np.rint(normalized * 255).astype(np.uint8)
    steps, points, probes = encoded.shape
    frames_by_probe = []
    means_by_probe = []
    peaks_by_probe = []
    for probe in range(probes):
        frames_by_probe.append(
            [base64.b64encode(encoded[step, :, probe].tobytes()).decode("ascii") for step in range(steps)]
        )
        means_by_probe.append([float(value) for value in values[:, :, probe].mean(axis=1)])
        peaks_by_probe.append([float(value) for value in values[:, :, probe].max(axis=1)])
    return {
        "encoding": "uint8-base64",
        "steps": int(steps),
        "pointCount": int(points),
        "probeCount": int(probes),
        "scaleQ99": scale,
        "framesByProbe": frames_by_probe,
        "meanAbsByProbe": means_by_probe,
        "peakAbsByProbe": peaks_by_probe,
    }


def summarize_activity(
    final_state: torch.Tensor,
    nodes_path: str | Path,
    *,
    seed: int = 20260912,
    spatial_points: int = 2_500,
    active_threshold: float = 0.1,
    spatial_indices: np.ndarray | None = None,
) -> dict:
    activity = final_state.detach().abs().mean(dim=1).float().cpu().numpy()
    nodes = feather.read_table(
        nodes_path,
        columns=["bodyId", "superclass", "somaLocation"],
        memory_map=True,
    )
    if len(nodes) != len(activity):
        raise ValueError("node annotations and model state have different lengths")
    body_ids = nodes["bodyId"].to_numpy(zero_copy_only=False)
    superclasses = np.asarray(nodes["superclass"].to_pylist(), dtype=object)
    groups: dict[str, list[float]] = defaultdict(list)
    for label, value in zip(superclasses, activity, strict=True):
        groups[str(label) if label is not None else "unassigned"].append(float(value))
    region_rows = []
    for label, values in groups.items():
        array = np.asarray(values, dtype=np.float32)
        region_rows.append(
            {
                "region": label,
                "neurons": int(len(array)),
                "meanAbs": float(array.mean()),
                "p95Abs": float(np.quantile(array, 0.95)),
                "activeFraction": float(np.mean(array > active_threshold)),
            }
        )
    region_rows.sort(key=lambda row: row["meanAbs"], reverse=True)

    locations = nodes["somaLocation"].to_pylist()
    valid = np.asarray([index for index, value in enumerate(locations) if value is not None], dtype=np.int64)
    if spatial_indices is None:
        rng = np.random.default_rng(seed)
        random_count = min(spatial_points, len(valid))
        chosen = rng.choice(valid, size=random_count, replace=False) if random_count else valid
        top_count = min(400, len(valid))
        valid_activity = activity[valid]
        top = valid[np.argpartition(valid_activity, -top_count)[-top_count:]] if top_count else valid
        chosen = np.unique(np.concatenate((chosen, top)))
        sampling = "deterministic random soma sample plus the 400 most active soma-bearing neurons"
    else:
        chosen = np.asarray(spatial_indices, dtype=np.int64)
        if chosen.ndim != 1 or np.any(chosen < 0) or np.any(chosen >= len(nodes)):
            raise ValueError("spatial indices must be valid node row indices")
        if any(locations[int(index)] is None for index in chosen):
            raise ValueError("spatial indices must refer to soma-bearing neurons")
        sampling = "stable deterministic soma sample shared by every replay"
    q99 = float(np.quantile(activity, 0.99)) if len(activity) else 1.0
    spatial = []
    for index in chosen:
        x, y, z = locations[int(index)]
        spatial.append(
            {
                "bodyId": int(body_ids[index]),
                "x": int(x),
                "y": int(y),
                "z": int(z),
                "activity": float(activity[index]),
                "normalized": float(min(1.0, activity[index] / max(q99, 1e-8))),
                "superclass": str(superclasses[index]),
            }
        )
    top_indices = np.argsort(activity)[-20:][::-1]
    return {
        "globalMeanAbs": float(activity.mean()),
        "globalPeakAbs": float(activity.max(initial=0.0)),
        "activeThreshold": active_threshold,
        "regions": region_rows,
        "spatial": spatial,
        "topNeurons": [
            {
                "bodyId": int(body_ids[index]),
                "activity": float(activity[index]),
                "superclass": str(superclasses[index]),
            }
            for index in top_indices
        ],
        "spatialSampling": sampling,
    }

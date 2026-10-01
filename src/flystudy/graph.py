from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.feather as feather
from scipy import sparse


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _column_name(names: list[str], *candidates: str) -> str:
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise ValueError(f"none of the required columns are present: {candidates}")


def _membership_positions(values: np.ndarray, sorted_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(sorted_ids, values)
    valid = positions < len(sorted_ids)
    valid_indices = np.flatnonzero(valid)
    valid[valid_indices] &= sorted_ids[positions[valid_indices]] == values[valid_indices]
    return positions, valid


def prepare_graph(
    annotations_path: str | Path,
    edges_path: str | Path,
    output_dir: str | Path,
    *,
    node_policy: str = "annotated",
) -> dict:
    """Build a postsynaptic-row graph without adding or thresholding edges."""
    annotations_path = Path(annotations_path)
    edges_path = Path(edges_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = feather.read_table(annotations_path, memory_map=True)
    if node_policy == "traced":
        mask = pc.fill_null(pc.equal(annotations["status"], "Traced"), False)
        selection = "status == 'Traced'"
    elif node_policy == "annotated":
        superclass_name = _column_name(annotations.column_names, "superclass", "super_class")
        mask = pc.is_valid(annotations[superclass_name])
        selection = f"{superclass_name} is assigned"
    else:
        raise ValueError("node_policy must be 'annotated' or 'traced'")

    nodes = annotations.filter(mask).sort_by([("bodyId", "ascending")])
    body_ids = nodes["bodyId"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    if len(body_ids) != len(np.unique(body_ids)):
        raise ValueError("selected annotation body IDs are not unique")

    edge_table = feather.read_table(edges_path, memory_map=True)
    names = edge_table.column_names
    pre_name = _column_name(names, "body_pre", "pre")
    post_name = _column_name(names, "body_post", "post")
    weight_name = _column_name(names, "weight", "count")
    selected_pre: list[np.ndarray] = []
    selected_post: list[np.ndarray] = []
    selected_weights: list[np.ndarray] = []
    raw_rows = 0
    for batch in edge_table.select([pre_name, post_name, weight_name]).to_batches(max_chunksize=2_000_000):
        pre = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        post = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        weights = batch.column(2).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        raw_rows += len(pre)
        pre_positions, pre_valid = _membership_positions(pre, body_ids)
        post_positions, post_valid = _membership_positions(post, body_ids)
        keep = pre_valid & post_valid
        if keep.any():
            selected_pre.append(pre_positions[keep].astype(np.int32))
            selected_post.append(post_positions[keep].astype(np.int32))
            selected_weights.append(weights[keep])

    if not selected_weights:
        raise ValueError("no released edges connect the selected neurons")
    pre_indices = np.concatenate(selected_pre)
    post_indices = np.concatenate(selected_post)
    contact_counts = np.concatenate(selected_weights)
    matrix = sparse.csr_array(
        (contact_counts, (post_indices, pre_indices)),
        shape=(len(body_ids), len(body_ids)),
        dtype=np.int64,
    )
    matrix.sum_duplicates()
    matrix.sort_indices()
    if matrix.data.max(initial=0) > np.iinfo(np.int32).max:
        raise OverflowError("aggregated contact count exceeds int32")
    crow = matrix.indptr.astype(np.int64, copy=False)
    cols = matrix.indices.astype(np.int32, copy=False)
    counts = matrix.data.astype(np.int32, copy=False)
    rows = np.repeat(np.arange(len(body_ids), dtype=np.int32), np.diff(crow))

    graph_path = output_dir / "graph.npz"
    np.savez(
        graph_path,
        body_ids=body_ids,
        crow=crow,
        rows=rows,
        cols=cols,
        counts=counts,
    )
    feather.write_feather(nodes, output_dir / "nodes.feather")
    manifest = {
        "dataset": "MaleCNS v1.0",
        "node_policy": node_policy,
        "selection": selection,
        "published_neuron_count": 166_700,
        "published_connected_neuron_count": 166_483,
        "neurons": int(len(body_ids)),
        "raw_connection_rows": int(raw_rows),
        "selected_connection_rows_before_aggregation": int(len(contact_counts)),
        "edges": int(matrix.nnz),
        "synaptic_contacts": int(counts.astype(np.int64).sum()),
        "self_edges": int(np.count_nonzero(rows == cols)),
        "orientation": "rows are postsynaptic; cols are presynaptic",
        "edge_threshold": None,
        "annotations_sha256": sha256_file(annotations_path),
        "edges_sha256": sha256_file(edges_path),
        "graph_sha256": sha256_file(graph_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from flystudy.graph import prepare_graph


def test_graph_preparation_filters_nodes_aggregates_edges_and_preserves_orientation(tmp_path: Path):
    annotations = pa.table(
        {
            "bodyId": pa.array([10, 20, 30, 40], type=pa.int64()),
            "status": ["Traced", "Traced", "Traced", "Glia"],
            "super_class": ["central", "central", "motor", "glia"],
            "class": ["A", "B", "C", "glia"],
            "side": ["L", "R", "M", "M"],
        }
    )
    edges = pa.table(
        {
            "body_pre": pa.array([10, 10, 20, 40], type=pa.int64()),
            "body_post": pa.array([20, 20, 30, 10], type=pa.int64()),
            "weight": pa.array([2, 3, 4, 99], type=pa.int64()),
        }
    )
    annotations_path = tmp_path / "annotations.feather"
    edges_path = tmp_path / "edges.feather"
    feather.write_feather(annotations, annotations_path)
    feather.write_feather(edges, edges_path)

    output = tmp_path / "graph"
    manifest = prepare_graph(annotations_path, edges_path, output, node_policy="traced")
    arrays = np.load(output / "graph.npz")
    assert manifest["neurons"] == 3
    assert manifest["edges"] == 2
    assert arrays["body_ids"].tolist() == [10, 20, 30]
    assert arrays["rows"].tolist() == [1, 2]
    assert arrays["cols"].tolist() == [0, 1]
    assert arrays["counts"].tolist() == [5, 4]
    assert json.loads((output / "manifest.json").read_text())["node_policy"] == "traced"


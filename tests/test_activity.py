import base64

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import torch

from flystudy.activity import encode_activity_replay, spatial_metadata, stable_spatial_indices, summarize_activity


def _nodes_file(tmp_path):
    path = tmp_path / "nodes.feather"
    table = pa.table(
        {
            "bodyId": [100, 101, 102, 103, 104],
            "superclass": ["a", "a", "b", "b", "c"],
            "somaLocation": [[1, 2, 3], None, [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        }
    )
    feather.write_feather(table, path)
    return path


def test_stable_spatial_indices_are_reproducible_and_soma_only(tmp_path):
    path = _nodes_file(tmp_path)
    first = stable_spatial_indices(path, seed=7, spatial_points=3)
    second = stable_spatial_indices(path, seed=7, spatial_points=3)
    assert np.array_equal(first, second)
    assert len(first) == 3
    assert 1 not in first


def test_activity_replay_has_one_aligned_byte_per_point(tmp_path):
    selected = torch.tensor(
        [
            [[0.0, 0.2], [0.5, 0.4], [1.0, 0.1]],
            [[0.2, 0.3], [0.6, 0.8], [0.9, 0.0]],
        ]
    )
    replay = encode_activity_replay(selected)
    assert replay["steps"] == 2
    assert replay["pointCount"] == 3
    assert replay["probeCount"] == 2
    for probe_frames in replay["framesByProbe"]:
        assert len(probe_frames) == 2
        assert all(len(base64.b64decode(frame)) == 3 for frame in probe_frames)

    path = _nodes_file(tmp_path)
    spatial = np.asarray([0, 2, 4])
    summary = summarize_activity(torch.ones(5, 2), path, spatial_indices=spatial)
    assert [row["bodyId"] for row in summary["spatial"]] == [100, 102, 104]
    assert summary["spatialSampling"].startswith("stable")
    metadata = spatial_metadata(path, spatial)
    assert [row["bodyId"] for row in metadata] == [100, 102, 104]
    assert all(row["activity"] == 0 for row in metadata)

"""Audited MaleCNS leg-afferent populations used as direct sensory inputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from flystudy.language_data import KINDS, LOCATIONS


LEG_NERVES = {"front": "ProLN", "middle": "MesoLN", "hind": "MetaLN"}
MODALITY_CRITERIA = {
    "tactile": ("mechanosensory_tactile", "mechanosensory bristle"),
    "gustatory": ("gustatory", "leg bristle"),
    "chordotonal": ("mechanosensory_proprioceptive", "chordotonal organ"),
    "campaniform": ("mechanosensory_proprioceptive", "campaniform sensilla"),
}


def anatomical_sensory_groups(graph_dir: str | Path) -> tuple[list[np.ndarray], list[dict]]:
    directory = Path(graph_dir)
    graph = np.load(directory / "graph.npz")
    graph_index = {int(body_id): index for index, body_id in enumerate(graph["body_ids"])}
    columns = ["bodyId", "superclass", "class", "subclass", "rootSide", "entryNerve"]
    table = feather.read_table(directory / "nodes.feather", columns=columns, memory_map=True)
    values = {name: table[name].to_pylist() for name in columns}
    groups, metadata, seen = [], [], set()
    for location, (side_word, segment) in enumerate(LOCATIONS):
        side = side_word[0].upper()
        nerve = LEG_NERVES[segment]
        for kind, name in enumerate(KINDS):
            sensory_class, sensory_subclass = MODALITY_CRITERIA[name]
            body_ids = []
            for row in range(len(table)):
                body_id = int(values["bodyId"][row])
                superclass = values["superclass"][row]
                if (body_id in graph_index and superclass is not None and "sensory" in superclass
                        and values["rootSide"][row] == side and values["entryNerve"][row] == nerve
                        and values["class"][row] == sensory_class and values["subclass"][row] == sensory_subclass):
                    body_ids.append(body_id)
            indices = np.asarray(sorted(graph_index[body_id] for body_id in body_ids), dtype=np.int64)
            if not len(indices) or len(indices) != len(set(body_ids)):
                raise RuntimeError(f"missing or duplicate anatomical group {side_word}/{segment}/{name}")
            overlap = seen.intersection(indices.tolist())
            if overlap:
                raise RuntimeError("anatomical input populations must be disjoint")
            seen.update(indices.tolist())
            groups.append(indices)
            metadata.append({"group": location * len(KINDS) + kind, "location": location, "kind": kind,
                             "side": side_word, "segment": segment, "entryNerve": nerve, "modality": name,
                             "class": sensory_class, "subclass": sensory_subclass,
                             "neurons": len(indices), "bodyIds": sorted(body_ids),
                             "perNeuronScale": float(1 / np.sqrt(len(indices)))})
    if len(groups) != len(LOCATIONS) * len(KINDS):
        raise RuntimeError("incomplete anatomical interface")
    return groups, metadata

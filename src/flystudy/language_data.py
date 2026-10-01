"""Anatomical location and sensory-population labels."""
from __future__ import annotations


LOCATIONS = [(side, part) for side in ("left", "right") for part in ("front", "middle", "hind")]


KINDS = "tactile gustatory chordotonal campaniform".split()

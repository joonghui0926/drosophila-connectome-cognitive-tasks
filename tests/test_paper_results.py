import importlib.util
from pathlib import Path

import numpy as np


def load_result_script():
    path = Path(__file__).parents[1] / "scripts" / "reproduce_paper_results.py"
    spec = importlib.util.spec_from_file_location("reproduce_paper_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_saved_records_reproduce_headline_values():
    result = load_result_script().calculate()
    addition = result["addition"]
    language = result["language"]

    assert round(100 * addition["anatomicalMean"], 2) == 92.77
    assert round(100 * addition["rewiredMean"], 2) == 67.93
    assert round(100 * language["factorial"]["anatomicalMean"], 2) == 61.59
    assert round(100 * language["factorial"]["rewiredMean"], 2) == 44.17
    assert round(100 * language["factorial"]["matchedNormalAnatomical"], 2) == 60.94
    assert round(100 * language["factorial"]["matchedShuffleAnatomical"], 2) == 19.27
    assert round(100 * language["nullRank"]["anatomical"], 2) == 71.35
    assert language["nullRank"]["rewiresAtOrAboveAnatomical"] == 0
    assert np.isclose(language["nullRank"]["oneSidedRankFraction"], 1 / 21)

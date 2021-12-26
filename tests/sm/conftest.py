from typing import List

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def time_series1() -> list:
    return pd.read_csv("examples/statistics/data/time_series1.txt", header=None)[0].tolist()


@pytest.fixture(scope="module")
def time_series2() -> list:
    return pd.read_csv("examples/statistics/data/time_series2.txt", header=None)[0].tolist()

@pytest.fixture(scope="module")
def dist_estimation_parameters() -> List[str]:
    return ["mle", "lmoments"]

@pytest.fixture(scope="module")
def dist_estimation_parameters_ks() -> str:
    return "lmoments"

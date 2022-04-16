from typing import List

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def cmap() -> str:
    return "terrain"


@pytest.fixture(scope="module")
def ColorScale() -> List[int]:
    return [1, 2, 3, 4, 5]

@pytest.fixture(scope="module")
def TicksSpacing() -> int:
    return 500

@pytest.fixture(scope="module")
def color_scale_2_gamma() -> float:
    return 0.5


@pytest.fixture(scope="module")
def color_scale_3_linscale() -> float:
    return 0.001


@pytest.fixture(scope="module")
def color_scale_3_linthresh() -> float:
    return 0.0001


@pytest.fixture(scope="module")
def midpoint() -> int:
    return 20

@pytest.fixture(scope="module")
def display_cellvalue() -> bool:
    return True


@pytest.fixture(scope="module")
def NumSize() -> int:
    return 8


@pytest.fixture(scope="module")
def Backgroundcolorthreshold():
    return None

@pytest.fixture(scope="module")
def points() -> pd.DataFrame:
    return pd.read_csv("examples/GIS/data/points.csv")

@pytest.fixture(scope="module")
def IDsize() -> int:
    return 20

@pytest.fixture(scope="module")
def IDcolor() -> str:
    return "green"

@pytest.fixture(scope="module")
def Gaugesize() -> int:
    return 100


@pytest.fixture(scope="module")
def Gaugecolor() -> str:
    return "blue"

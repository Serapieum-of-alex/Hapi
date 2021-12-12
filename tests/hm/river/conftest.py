from typing import List
import pytest

@pytest.fixture(scope="module")
def slope_path() -> str:
    return "Examples/Hydrodynamic models/test_case/inputs/1d/topo/slope.csv"


@pytest.fixture(scope="module")
def river_cross_section_path() -> str:
    return "Examples/Hydrodynamic models/test_case/inputs/1d/topo/xs_same_downward-3segment.csv"


@pytest.fixture(scope="module")
def river_network_path() -> str:
    return "Examples/Hydrodynamic models/test_case/inputs/1d/topo/rivernetwork-3segments.txt"

@pytest.fixture(scope="module")
def create_sub_instance_subid() -> int:
    return 1

@pytest.fixture(scope="module")
def create_sub_instance_firstxs() -> int:
    return 1

@pytest.fixture(scope="module")
def test_sub_GetFlow_lateralTable() -> List[int]:
    return [30, 50, 70]


@pytest.fixture(scope="module")
def create_sub_instance_lastxs() -> int:
    return 100

@pytest.fixture(scope="module")
def test_Read1DResult_path() -> str:
    return "Examples/Hydrodynamic models/test_case/results/1d/"

@pytest.fixture(scope="module")
def test_Read1DResult_xsid() -> int:
    return 50
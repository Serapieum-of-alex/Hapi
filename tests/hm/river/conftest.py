from typing import List

import pytest

time_series_length = 80
laterals = [30, 50, 70]
sub_id = 1
sub_id_us = 3
segment3_specificxs = 270
us_subs = [1,2]
first_xs = 1
last_xs = 100
xsid = 50
station_id = 999666
date_format = "'%Y-%m-%d'"

@pytest.fixture(scope="module")
def test_time_series_length() -> int:
    return time_series_length

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
def segment1() -> int:
    return sub_id

@pytest.fixture(scope="module")
def segment3() -> int:
    return sub_id_us

@pytest.fixture(scope="module")
def segment3_us_subs() -> List[int]:
    return us_subs

@pytest.fixture(scope="module")
def segment3_specificxs_plot() -> int:
    return segment3_specificxs

@pytest.fixture(scope="module")
def create_sub_instance_firstxs() -> int:
    return first_xs

@pytest.fixture(scope="module")
def sub_GetFlow_lateralTable() -> List[int]:
    return laterals


@pytest.fixture(scope="module")
def create_sub_instance_lastxs() -> int:
    return last_xs

@pytest.fixture(scope="module")
def Read1DResult_path() -> str:
    return "Examples/Hydrodynamic models/test_case/results/1d/"

@pytest.fixture(scope="module")
def Read1DResult_xsid() -> int:
    return xsid


@pytest.fixture(scope="module")
def ReadRRMHydrograph_station_id() -> int:
    return station_id


@pytest.fixture(scope="module")
def ReadRRMHydrograph_date_format() -> str:
    return date_format


@pytest.fixture(scope="module")
def ReadRRMHydrograph_location_1() -> int:
    return 1

@pytest.fixture(scope="module")
def ReadRRMHydrograph_location_2() -> int:
    return 2

@pytest.fixture(scope="module")
def ReadRRMHydrograph_path() -> str:
    return "Examples/Hydrodynamic models/test_case/inputs/rrm/rrm_location"


@pytest.fixture(scope="module")
def ReadRRMHydrograph_path2() -> str:
    return "Examples/Hydrodynamic models/test_case/inputs/rrm/hm_location"


@pytest.fixture(scope="module")
def CustomizedRunspath() -> str:
    return "Examples/Hydrodynamic models/test_case/results/customized_results/"

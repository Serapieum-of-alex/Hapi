from typing import List

import pytest

segment1_laterals = [30, 50, 70]
sub_id = 1
sub_id_us = 3
segment3_lastsegment = True
segment3_specificxs = 270
segment3_xs_ids = list(range(201, 301))
us_subs = [1, 2]
first_xs = 1
last_xs = 100
xsid = 50
station_id = 999666
date_format = "'%Y-%m-%d'"
SP_columns = [
    "RP2",
    "RP5",
    "RP10",
    "RP15",
    "RP20",
    "RP50",
    "RP100",
    "RP200",
    "RP500",
    "RP1000",
    "RP5000",
]


@pytest.fixture(scope="module")
def slope_path() -> str:
    return "examples/hydrodynamic-models/test_case/inputs/1d/topo/slope.csv"


@pytest.fixture(scope="module")
def river_cross_section_path() -> str:
    return "examples/hydrodynamic-models/test_case/inputs/1d/topo/xs_same_downward-3segment.csv"


@pytest.fixture(scope="module")
def river_network_path() -> str:
    return "examples/hydrodynamic-models/test_case/inputs/1d/topo/rivernetwork-3segments.txt"


@pytest.fixture(scope="module")
def distribution_properties_fpath() -> str:
    return "examples/hydrodynamic-models/test_case/inputs/gauges/discharge_long_ts/statistical-analysis-results/distribution-properties.csv"


@pytest.fixture(scope="module")
def distribution_properties_hm_results_fpath() -> str:
    return "examples/hydrodynamic-models/test_case/results/customized_results/discharge_long_ts/statistical-analysis-results/distribution-properties.csv"


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
def segment3_xs_ids_list() -> list:
    return segment3_xs_ids


@pytest.fixture(scope="module")
def create_sub_instance_firstxs() -> int:
    return first_xs


@pytest.fixture(scope="module")
def sub_GetFlow_lateralTable() -> List[int]:
    return segment1_laterals


@pytest.fixture(scope="module")
def create_sub_instance_lastxs() -> int:
    return last_xs


@pytest.fixture(scope="module")
def segment1_xs() -> int:
    return 50


@pytest.fixture(scope="module")
def segment3_xs() -> int:
    return 250


@pytest.fixture(scope="module")
def Read1DResult_path() -> str:
    return "examples/hydrodynamic-models/test_case/results/1d/"


@pytest.fixture(scope="module")
def usbc_path() -> str:
    return "examples/hydrodynamic-models/test_case/results/USbnd/"


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
    return "examples/hydrodynamic-models/test_case/inputs/rrm/rrm_location"


@pytest.fixture(scope="module")
def ReadRRMHydrograph_path2() -> str:
    return "examples/hydrodynamic-models/test_case/inputs/rrm/hm_location"


@pytest.fixture(scope="module")
def CustomizedRunspath() -> str:
    return "examples/hydrodynamic-models/test_case/results/customized_results/"


@pytest.fixture(scope="module")
def lastsegment() -> bool:
    return segment3_lastsegment


@pytest.fixture(scope="module")
def subdailyresults_path() -> str:
    return "examples/hydrodynamic-models/test_case/results/"


@pytest.fixture(scope="module")
def subdaily_no_timesteps() -> int:
    return 24 * 60


@pytest.fixture(scope="module")
def onemin_results_dates() -> list:
    start = "1955-01-01"
    end = "1955-01-10"
    return [start, end]


@pytest.fixture(scope="module")
def onemin_days() -> int:
    return 9


@pytest.fixture(scope="module")
def onemin_results_len() -> int:
    return 10


@pytest.fixture(scope="module")
def statistical_properties_columns() -> list:
    return SP_columns


@pytest.fixture(scope="module")
def overtopping_files_dir() -> str:
    return "tests/hm/data/overtopping_files"


@pytest.fixture(scope="module")
def event_index_file() -> str:
    return "tests/hm/data/event-index.txt"

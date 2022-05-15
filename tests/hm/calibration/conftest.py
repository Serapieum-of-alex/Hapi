from typing import List

import pytest


# @pytest.fixture(scope="module")
# def gauges_file_extension() -> str:
    # return ".csv"


@pytest.fixture(scope="module")
def gauge_date_format() -> str:
    return "'%Y-%m-%d'"


@pytest.fixture(scope="module")
def gauge_long_ts_date_format() -> str:
    return "%Y-%m-%d"

@pytest.fixture(scope="module")
def gauges_table_path() -> str:
    return "examples/Hydrodynamic models/test_case/inputs/gauges/gauges.geojson"


@pytest.fixture(scope="module")
def gauges_numbers() -> int:
    return 3

@pytest.fixture(scope="module")
def ReadObservedQ_Path() -> str:
    return "examples/Hydrodynamic models/test_case/inputs/gauges/discharge/"


@pytest.fixture(scope="module")
def ObservedQ_long_ts_Path() -> str:
    return "examples/Hydrodynamic models/test_case/inputs/gauges/discharge_long_ts/"


@pytest.fixture(scope="module")
def ObservedQ_long_ts_dates() -> List[str]:
    return ["1951-01-01", "2003-12-31"]

@pytest.fixture(scope="module")
def ObservedQ_long_ts_len() -> int:
    return 54


@pytest.fixture(scope="module")
def ReadObservedWL_Path() -> str:
    return "examples/Hydrodynamic models/test_case/inputs/gauges/water_level/"

@pytest.fixture(scope="module")
def calibrateProfile_DS_bedlevel() -> float:
    return 61

@pytest.fixture(scope="module")
def calibrateProfile_mn() -> float:
    return 0.06

@pytest.fixture(scope="module")
def calibrateProfile_slope() -> float:
    return -0.03

@pytest.fixture(scope="module")
def DownWardBedLevel_height() -> float:
    return 0.05


@pytest.fixture(scope="module")
def rrmpath() -> str:
    return r"examples\Hydrodynamic models\test_case\inputs\rrm\rrm_location"


@pytest.fixture(scope="module")
def rrmpath_long_ts() -> str:
    return r"examples\Hydrodynamic models\test_case\inputs\rrm\long_ts\rrm_location"


@pytest.fixture(scope="module")
def rrm_long_ts_number() -> int:
    return 61


@pytest.fixture(scope="module")
def rrmgauges() -> List[int]:
    return [444222, 888555, 999666]


@pytest.fixture(scope="module")
def hm_separated_q_results_path() -> str:
    return r"examples\Hydrodynamic models\test_case\results\separated-results\discharge"

@pytest.fixture(scope="module")
def hm_separated_wl_results_path() -> str:
    return r"examples\Hydrodynamic models\test_case\results\separated-results\water_level"


@pytest.fixture(scope="module")
def hm_separated_results_q_long_ts_path() -> str:
    return r"examples\Hydrodynamic models\test_case\results\separated-results\discharge\long_ts"


@pytest.fixture(scope="module")
def hm_long_ts_number() -> int:
    return 61

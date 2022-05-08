import pytest


@pytest.fixture(scope="module")
def gauges_file_extension() -> str:
    return ".csv"


@pytest.fixture(scope="module")
def gauge_date_format() -> str:
    return "'%Y-%m-%d'"


@pytest.fixture(scope="module")
def gauges_table_path() -> str:
    return "examples/Hydrodynamic models/test_case/inputs/gauges/gauges.geojson"


@pytest.fixture(scope="module")
def ReadObservedQ_Path() -> str:
    return "examples/Hydrodynamic models/test_case/inputs/gauges/discharge/"


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
def rrmgauges() -> str:
    return [444222, 888555, 999666]


@pytest.fixture(scope="module")
def hm_separated_results_path() -> str:
    return r"examples\Hydrodynamic models\test_case\results\separated-results"
import pytest

@pytest.fixture(scope="module")
def calibration_gauges_file_extension() -> str:
    return ".csv"


@pytest.fixture(scope="module")
def gauge_date_format() -> str:
    return "'%Y-%m-%d'"


@pytest.fixture(scope="module")
def calibration_gauges_table_path() -> str:
    return "Examples/Hydrodynamic models/test_case/inputs/gauges/gauges.csv"


@pytest.fixture(scope="module")
def calibration_ReadObservedQ_Path() -> str:
    return "Examples/Hydrodynamic models/test_case/inputs/gauges/discharge/"


@pytest.fixture(scope="module")
def calibration_ReadObservedWL_Path() -> str:
    return "Examples/Hydrodynamic models/test_case/inputs/gauges/water_level/"
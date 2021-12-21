import pytest

from tests.hm.calibration.conftest import *
from tests.hm.inputs.conftest import *
from tests.hm.interface.conftest import *
from tests.hm.river.conftest import *

time_series_length = 80
hours = list(range(1,25))

@pytest.fixture(scope="module")
def version() -> int:
    return 3

@pytest.fixture(scope="module")
def dates() -> list:
    start = "1955-01-01"
    end = "1955-03-21"
    return [start, end]

@pytest.fixture(scope="module")
def rrm_start() -> str:
    return "1955-1-1"

@pytest.fixture(scope="module")
def nodatavalu() -> int:
    return -9

@pytest.fixture(scope="module")
def xs_total_no() -> int:
    return 300

@pytest.fixture(scope="module")
def xs_col_no() -> int:
    return 16

@pytest.fixture(scope="module")
def test_time_series_length() -> int:
    return time_series_length

@pytest.fixture(scope="module")
def test_hours() -> list:
    return hours

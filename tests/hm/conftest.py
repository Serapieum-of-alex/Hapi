import pytest
from tests.hm.river.conftest import *
from tests.hm.interface.conftest import *
from tests.hm.calibration.conftest import *

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
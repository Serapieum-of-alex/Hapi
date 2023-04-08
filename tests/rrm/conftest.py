from tests.rrm.calibration.conftest import *
from tests.rrm.catchment.conftest import *


@pytest.fixture(scope="module")
def coello_start_date() -> str:
    return "2009-01-01"

@pytest.fixture(scope="module")
def coello_end_date() -> str:
    return "2009-01-11"
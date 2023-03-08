import pytest

from tests.hm.calibration.conftest import *
from tests.hm.inputs.conftest import *
from tests.hm.interface.conftest import *
from tests.hm.river.conftest import *

time_series_length = 80
hours = list(range(1, 25))


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
    return 17


@pytest.fixture(scope="module")
def test_time_series_length() -> int:
    return time_series_length


@pytest.fixture(scope="module")
def test_hours() -> list:
    return hours


@pytest.fixture(scope="module")
def combine_rdir() -> str:
    return "tests/hm/data/results/combin_results"


@pytest.fixture(scope="module")
def combine_save_to() -> str:
    return "tests/hm/data/results/combin_results/combined"


@pytest.fixture(scope="module")
def separated_folders() -> List[str]:
    return ["1d(1-5)", "1d(6-10)"]


@pytest.fixture(scope="module")
def separated_folders_file_names() -> List[str]:
    return ["1.txt", "1_left.txt", "1_right.txt"]


@pytest.fixture(scope="module")
def overtopping_file() -> str:
    return "tests/hm/data/overtopping.txt"


@pytest.fixture(scope="module")
def event_instance_attrs() -> List[str]:
    return [
        "left_overtopping_suffix",
        "right_overtopping_suffix",
        "duration_prefix",
        "return_period_prefix",
        "two_d_result_path",
        "compressed",
        "extracted_values",
        "event_index",
    ]

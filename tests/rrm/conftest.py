from typing import Dict, Tuple
import numpy as np
from tests.rrm.calibration.conftest import *
from tests.rrm.catchment.conftest import *


@pytest.fixture(scope="module")
def coello_start_date() -> str:
    return "2009-01-01"


@pytest.fixture(scope="module")
def coello_end_date() -> str:
    return "2009-01-11"


@pytest.fixture(scope="module")
def coello_evap_path() -> str:
    return "tests/rrm/data/evap"


@pytest.fixture(scope="module")
def coello_prec_path() -> str:
    return "tests/rrm/data/prec"


@pytest.fixture(scope="module")
def coello_temp_path() -> str:
    return "tests/rrm/data/temp"


@pytest.fixture(scope="module")
def coello_acc_path() -> str:
    return "tests/rrm/data/gis/acc4000.tif"


@pytest.fixture(scope="module")
def coello_acc_values() -> List:
    return [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        10,
        11,
        13,
        15,
        16,
        17,
        23,
        43,
        44,
        48,
        55,
        59,
        63,
        86,
        88,
    ]


@pytest.fixture(scope="module")
def coello_fd_path() -> str:
    return "tests/rrm/data/gis/fd4000.tif"


@pytest.fixture(scope="module")
def coello_fdt() -> Dict:
    return {
        "1,5": [],
        "1,6": [],
        "1,7": [],
        "2,5": [(1, 5)],
        "2,6": [],
        "2,7": [(1, 6), (1, 7)],
        "3,3": [],
        "3,4": [],
        "3,5": [(2, 5)],
        "3,6": [],
        "3,7": [(2, 6), (2, 7)],
        "3,8": [],
        "3,9": [],
        "4,3": [],
        "4,4": [(3, 3), (3, 4), (4, 3), (5, 3)],
        "4,5": [(3, 5), (3, 6)],
        "4,6": [],
        "4,7": [(3, 7)],
        "4,8": [(3, 8), (3, 9)],
        "4,9": [],
        "5,3": [],
        "5,4": [],
        "5,5": [(4, 4), (4, 5), (4, 6)],
        "5,6": [],
        "5,7": [],
        "5,8": [(4, 7), (4, 8), (5, 7)],
        "5,9": [(4, 9)],
        "6,2": [],
        "6,3": [],
        "6,4": [],
        "6,5": [(5, 4), (5, 5), (5, 6), (6, 4)],
        "6,6": [],
        "6,7": [],
        "6,8": [],
        "6,9": [(5, 8), (5, 9)],
        "7,1": [],
        "7,2": [(7, 1)],
        "7,3": [(6, 2)],
        "7,4": [(6, 3), (7, 3), (8, 3)],
        "7,5": [(7, 4), (8, 4)],
        "7,6": [(6, 5), (6, 6), (7, 5), (8, 5), (8, 6)],
        "7,7": [(6, 7)],
        "7,8": [],
        "7,9": [(6, 8), (6, 9)],
        "8,1": [],
        "8,2": [(8, 1)],
        "8,3": [(7, 2), (8, 2), (9, 2)],
        "8,4": [(9, 3), (9, 4)],
        "8,5": [],
        "8,6": [],
        "8,7": [(7, 6)],
        "8,8": [(7, 7), (7, 8), (8, 7)],
        "8,9": [],
        "8,10": [(7, 9)],
        "8,11": [(8, 10)],
        "8,12": [],
        "9,1": [],
        "9,2": [(9, 1), (10, 1)],
        "9,3": [(10, 2), (10, 3)],
        "9,4": [],
        "9,5": [(10, 4)],
        "9,6": [(9, 5), (10, 5)],
        "9,7": [(9, 6)],
        "9,8": [(9, 7)],
        "9,9": [(8, 8), (9, 8)],
        "9,10": [(8, 9)],
        "9,11": [(9, 10)],
        "9,12": [(8, 11), (8, 12), (9, 11), (10, 11)],
        "10,0": [],
        "10,1": [(10, 0), (11, 0), (11, 1)],
        "10,2": [(11, 2)],
        "10,3": [],
        "10,4": [],
        "10,5": [],
        "10,7": [],
        "10,8": [(10, 7)],
        "10,9": [(10, 8)],
        "10,10": [(9, 9), (10, 9)],
        "10,11": [(10, 10), (11, 10), (11, 11)],
        "10,12": [],
        "10,13": [(9, 12), (10, 12)],
        "11,0": [],
        "11,1": [(12, 1)],
        "11,2": [(12, 2)],
        "11,9": [],
        "11,10": [(11, 9)],
        "11,11": [],
        "12,1": [],
        "12,2": [],
    }


@pytest.fixture(scope="module")
def coello_cat_area() -> int:
    return 1530


@pytest.fixture(scope="module")
def coello_initial_cond() -> List:
    return [0, 5, 5, 5, 0]


@pytest.fixture(scope="module")
def coello_parameter_bounds() -> Tuple[List, List]:
    UB = np.loadtxt("tests/rrm/data/calibration/UB-tot.txt", usecols=0)
    LB = np.loadtxt("tests/rrm/data/calibration/LB-tot.txt", usecols=0)
    return LB, UB


@pytest.fixture(scope="module")
def coello_gauges_table() -> str:
    return "tests/rrm/data/calibration/gauges.csv"

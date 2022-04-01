import pytest
import numpy as np
# import datetime as dt
from osgeo import gdal
from osgeo.gdal import Dataset


@pytest.fixture(scope="module")
def src() -> Dataset:
    return gdal.Open("examples/GIS/data/acc4000.tif")


@pytest.fixture(scope="module")
def src_no_data_value() -> float:
    return -3.402823e+38


@pytest.fixture(scope="module")
def src_epsg() -> int:
    return 32618

@pytest.fixture(scope="module")
def src_geotransform() -> tuple:
    return (432968.1206170588, 4000.0, 0.0, 520007.787999178, 0.0, -4000.0)


@pytest.fixture(scope="module")
def src_arr_first_4_rows() -> np.ndarray:
    return np.array([[434968.12061706, 520007.78799918],
                     [434968.12061706, 520007.78799918],
                     [434968.12061706, 520007.78799918],
                     [434968.12061706, 520007.78799918]])


@pytest.fixture(scope="module")
def src_arr_last_4_rows() -> np.ndarray:
    return np.array([[478968.12061706, 520007.78799918],
                     [478968.12061706, 520007.78799918],
                     [478968.12061706, 520007.78799918],
                     [478968.12061706, 520007.78799918]])


@pytest.fixture(scope="module")
def cells_centerscoords() -> np.ndarray:
    return np.array([[434968.12061706, 520007.78799918],
                    [438968.12061706, 520007.78799918],
                    [442968.12061706, 520007.78799918]])


@pytest.fixture(scope="module")
def rasters_folder_path() -> str:
    return "examples/GIS/data/raster-folder"


@pytest.fixture(scope="module")
def rasters_folder_rasters_number() -> int:
    return 6


@pytest.fixture(scope="module")
def rasters_folder_dim() -> tuple:
    return 125, 93


@pytest.fixture(scope="module")
def rasters_folder_start_date() -> str:
    return "1979-01-02"


@pytest.fixture(scope="module")
def rasters_folder_end_date() -> str:
    return "1979-01-05"


@pytest.fixture(scope="module")
def rasters_folder_date_fmt() -> str:
    return "%Y-%m-%d"


@pytest.fixture(scope="module")
def rasters_folder_between_dates_raster_number() -> int:
    return 4
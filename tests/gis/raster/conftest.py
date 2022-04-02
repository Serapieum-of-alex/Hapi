import pytest
import numpy as np
# import datetime as dt
from osgeo import gdal
from osgeo.gdal import Dataset


@pytest.fixture(scope="module")
def src() -> Dataset:
    return gdal.Open("examples/GIS/data/acc4000.tif")

@pytest.fixture(scope="module")
def src_arr(src: Dataset) -> np.ndarray:
    return src.ReadAsArray()

@pytest.fixture(scope="module")
def src_no_data_value() -> float:
    return -3.402823e+38


@pytest.fixture(scope="module")
def src_epsg() -> int:
    return 32618

@pytest.fixture(scope="module")
def src_geotransform() -> tuple:
    return 432968.1206170588, 4000.0, 0.0, 520007.787999178, 0.0, -4000.0


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
def save_raster_path() -> str:
    return "examples/GIS/data/save_raster_test.tif"


@pytest.fixture(scope="module")
def raster_like_path() -> str:
    return "examples/GIS/data/raster_like_saved.tif"


def func1(val):
    if val < 20:
        val = 1
    elif val < 40:
        val = 2
    elif val < 60:
        val = 3
    elif val < 80:
        val = 4
    elif val < 100:
        val = 5
    else:
        val = 0
    return val


@pytest.fixture(scope="module")
def mapalgebra_function():
    return func1


@pytest.fixture(scope="module")
def fill_raster_path() -> str:
    return "examples/GIS/data/fill_raster_saved.tif"


@pytest.fixture(scope="module")
def fill_raster_value() -> int:
    return 20


@pytest.fixture(scope="module")
def resample_raster_cell_size() -> int:
    return 100


@pytest.fixture(scope="module")
def resample_raster_resample_technique() -> str:
    return "bilinear"

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


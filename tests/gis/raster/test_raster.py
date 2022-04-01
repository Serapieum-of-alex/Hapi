import pytest
import numpy as np
from osgeo.gdal import Dataset
from Hapi.gis.raster import Raster

def test_GetRasterData(
        src: Dataset,
        src_no_data_value: float,
):
    arr, nodataval = Raster.GetRasterData(src)
    assert np.isclose(src_no_data_value, nodataval, rtol=0.001)
    assert isinstance(arr, np.ndarray)


def test_GetProjectionData(
        src: Dataset,
        src_epsg: int,
        src_geotransform: tuple,
):
    epsg, geo = Raster.GetProjectionData(src)
    assert epsg == src_epsg
    assert geo == src_geotransform


def test_GetCellCoords(
        src: Dataset,
        src_arr_first_4_rows: np.ndarray,
        src_arr_last_4_rows: np.ndarray,
        cells_centerscoords: np.ndarray,
):
    coords, centerscoords = Raster.GetCellCoords(src)
    assert np.isclose(coords[:4, :], src_arr_first_4_rows, rtol=0.000001).all()
    assert np.isclose(coords[-4:, :], src_arr_last_4_rows, rtol = 0.000001).all()
    assert np.isclose(centerscoords[0][:3], cells_centerscoords, rtol=0.000001).all()



class TestReadRastersFolder:
    def test_read_all_inside_folder_without_order(
            self,
            rasters_folder_path: str,
            rasters_folder_rasters_number: int,
            rasters_folder_dim: tuple
    ):
        arr = Raster.ReadRastersFolder(rasters_folder_path, WithOrder=False)
        assert np.shape(arr) == (rasters_folder_dim[0], rasters_folder_dim[1], rasters_folder_rasters_number)


    def test_read_all_inside_folder(
            self,
            rasters_folder_path: str,
            rasters_folder_rasters_number: int,
            rasters_folder_dim: tuple
    ):
        arr = Raster.ReadRastersFolder(rasters_folder_path, WithOrder=True)
        assert np.shape(arr) == (rasters_folder_dim[0], rasters_folder_dim[1], rasters_folder_rasters_number)


    def test_read_between_dates(
            self,
            rasters_folder_path: str,
            rasters_folder_start_date: str,
            rasters_folder_end_date: str,
            rasters_folder_date_fmt: str,
            rasters_folder_dim: tuple,
            rasters_folder_between_dates_raster_number: int
    ):
        arr = Raster.ReadRastersFolder(
            rasters_folder_path,
            WithOrder=True,
            start=rasters_folder_start_date,
            end=rasters_folder_end_date,
            fmt=rasters_folder_date_fmt
        )
        assert np.shape(arr) == (rasters_folder_dim[0], rasters_folder_dim[1], rasters_folder_between_dates_raster_number)



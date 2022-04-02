import os

import numpy as np
from osgeo import gdal
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


def test_create_raster(
        src_arr: np.ndarray,
        src_geotransform: tuple,
        src_epsg: int,
        src_no_data_value: float,
):
    src = Raster.CreateRaster(arr=src_arr,
                              geo=src_geotransform,
                              EPSG=src_epsg,
                              NoDataValue=src_no_data_value
                              )
    assert isinstance(src, Dataset)
    assert np.isclose(src.ReadAsArray(), src_arr, rtol=0.00001).all()
    assert np.isclose(src.GetRasterBand(1).GetNoDataValue(), src_no_data_value, rtol=0.00001)
    assert src.GetGeoTransform() == src_geotransform



def test_save_rasters(
        src: Dataset,
        save_raster_path: str,
):
    Raster.SaveRaster(src, save_raster_path)
    assert os.path.exists(save_raster_path)
    os.remove(save_raster_path)



def test_raster_like(
        src: Dataset,
        src_arr: np.ndarray,
        src_no_data_value: float,
        raster_like_path: str,
):
    arr2 = np.ones(shape=src_arr.shape, dtype=np.float64) * src_no_data_value
    arr2[~np.isclose(src_arr, src_no_data_value, rtol=0.001)] = 5

    Raster.RasterLike(src, arr2, raster_like_path)
    dst = gdal.Open(raster_like_path)
    arr = dst.ReadAsArray()
    assert arr.shape == src_arr.shape
    assert np.isclose(src.GetRasterBand(1).GetNoDataValue(), src_no_data_value, rtol=0.00001)
    assert src.GetGeoTransform() == dst.GetGeoTransform()
    os.path.exists(raster_like_path)

def test_map_algebra(
        src: Dataset,
        mapalgebra_function,
):
    dst = Raster.MapAlgebra(src, mapalgebra_function)
    arr = dst.ReadAsArray()
    nodataval = dst.GetRasterBand(1).GetNoDataValue()
    vals = arr[~np.isclose(arr, nodataval, rtol=0.00000000000001)]
    vals = list(set(vals))
    assert vals == [1, 2, 3, 4, 5]


def test_fill_raster(
        src: Dataset,
        fill_raster_path: str,
        fill_raster_value: int
):
    Raster.RasterFill(src, fill_raster_value, SaveTo=fill_raster_path)
    "now the resulted raster is saved to disk"
    dst = gdal.Open(fill_raster_path)
    arr = dst.ReadAsArray()
    nodataval = dst.GetRasterBand(1).GetNoDataValue()
    vals = arr[~np.isclose(arr, nodataval, rtol=0.00000000000001)]
    vals = list(set(vals))
    assert vals[0] == fill_raster_value


# def test_(
#         src: Dataset,
#         resample_raster_cell_size: int,
#         resample_raster_resample_technique: str,
# ):
#     dst = Raster.ResampleRaster(src,
#                                 resample_raster_cell_size,
#                                 resample_technique=resample_raster_resample_technique
#                                 )
#
#     dst_arr, _ = Raster.GetRasterData(dst)
#     _, newgeo = Raster.GetProjectionData(dst)

class TestReadRastersFolder:
    def test_read_all_inside_folder_without_order(
            self,
            rasters_folder_path: str,
            rasters_folder_rasters_number: int,
            rasters_folder_dim: tuple
    ):
        arr = Raster.ReadRastersFolder(rasters_folder_path, with_order=False)
        assert np.shape(arr) == (rasters_folder_dim[0], rasters_folder_dim[1], rasters_folder_rasters_number)


    def test_read_all_inside_folder(
            self,
            rasters_folder_path: str,
            rasters_folder_rasters_number: int,
            rasters_folder_dim: tuple
    ):
        arr = Raster.ReadRastersFolder(rasters_folder_path, with_order=True)
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
        arr = Raster.ReadRastersFolder(rasters_folder_path, with_order=True, start=rasters_folder_start_date,
                                       end=rasters_folder_end_date, fmt=rasters_folder_date_fmt)
        assert np.shape(arr) == (rasters_folder_dim[0],
                                 rasters_folder_dim[1],
                                 rasters_folder_between_dates_raster_number
                                 )

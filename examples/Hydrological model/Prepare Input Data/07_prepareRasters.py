"""Created on Sat Mar 27 19:09:20 2021.

@author: mofarrag

Make sure the working directory is set to the examples folder in the Hapi repo"
currunt_work_directory = Hapi/Example
"""
import gdal

from Hapi.raster import Raster

dem_path = "Data/GIS/Hapi_GIS_Data/acc4000.tif"
SaveTo = "data/parameters/"
#%%
"craeate a raster typicall to the DEM and fill it with 1"
K = 1
src = gdal.Open(dem_path)

Raster.RasterFill(src, K, SaveTo + "11_K_muskingum.tif")
#%%
X = 0.2

Raster.RasterFill(src, X, SaveTo + "12_X_muskingum.tif")

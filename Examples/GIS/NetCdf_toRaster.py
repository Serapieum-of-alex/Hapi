# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 22:30:48 2020

@author: mofarrag
https://www.linkedin.com/pulse/convert-netcdf4-file-geotiff-using-python-chonghua-yin/
"""
from IPython import get_ipython
get_ipython().magic("reset -f")
import os
import sys
from osgeo import gdal_array
import gdal
import osr
import xarray as xr
import numpy as np

import rasterio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%%
# FileName = "F:/01Algorithms/HAPI/Examples/data/GIS/mslp.mon.mean.nc"
# VarName = "mslp"
# SaveTo = "F:/01Algorithms/HAPI/Examples/data/GIS/" + VarName + ".tif"

FileName = "F:/01Algorithms/HAPI/Examples/data/GIS/MSWEP_1979010100.nc"
SaveTo = "F:/01Algorithms/HAPI/Examples/data/GIS/MSWEP_1979010100.tif"
VarName=''


# def NetCDFtoRaster(Filename, SaveTo, VarName=''):
#     """

#     ==================================================================
#         Function to read the original file's projection
#     ==================================================================
#     """

### Get netCDF Info
# open the netcdf file
src_ds = gdal.Open(FileName)

if src_ds is None:
    print("Open Failed")
    sys.exit()
# get the number of variables in the netCDF
VarNo = len(src_ds.GetSubDatasets())

if VarNo > 1:
    # if exists more than 1 var in the netCDF
    subdataset = 'NETCDF:"' + FileName + '":' + VarName
    # open the variable inside the dataset
    src_ds_sd = gdal.Open(subdataset)
    # begin to read info of the named variable
    NoValue = src_ds_sd.GetRasterBand(1).GetNoDataValue()
    src_col = src_ds_sd.RasterXSize
    src_row = src_ds_sd.RasterYSize
    src_gt = src_ds_sd.GetGeoTransform()
    src_sref = osr.SpatialReference()
    src_sref.ImportFromWkt(src_ds_sd.GetProjectionRef())
    # close the subdataset and the whole dataset
    src_ds_sd = None
    src_ds = None

    # read data using xarray
    xr_ensemble = xr.open_dataset(FileName)
    data = xr_ensemble[VarName]
    data = np.ma.masked_array(data, mask=data==NoValue, fill_value=NoValue)
else:
    # if exists more than 1 var in the netCDF
    subdataset = 'NETCDF:"' + FileName
    # open the variable inside the dataset
    src_ds_sd = gdal.Open(subdataset)
    # begin to read info of the named variable
    NoValue = src_ds_sd.GetRasterBand(1).GetNoDataValue()
    src_col = src_ds_sd.RasterXSize
    src_row = src_ds_sd.RasterYSize
    src_gt = src_ds_sd.GetGeoTransform()
    src_sref = osr.SpatialReference()
    src_sref.ImportFromWkt(src_ds_sd.GetProjectionRef())
    # close the subdataset and the whole dataset
    src_ds_sd = None
    src_ds = None

    # read data using xarray
    data = xr.open_dataset(FileName).to_array()
    data = np.ma.masked_array(data, mask=data==NoValue, fill_value=NoValue)



# create tif file
DataType = gdal_array.NumericTypeCodeToGDALTypeCode(data.dtype)

if type(DataType) != np.int:
    if DataType.startswith('gdal.GDT_') == False:
        DataType = eval('gdal.GDT_' + DataType)

# NewFileName = SaveTo + Varname + '.tif'

zsize = data.shape[0]

# create a driver
driver = gdal.GetDriverByName('GTiff')
# set nans to the original NO Data Values
data[np.isnan(data)] = NoValue
# set up the dataset with the zsize bands
DataSet = driver.Create(SaveTo, src_col, src_row, zsize, DataType)
DataSet.SetGeoTransform(src_gt)
DataSet.SetProjection(src_sref.ExportToWkt())
# write each slice of the array along the zsize
if VarNo > 1:
    for i in range(0,zsize):
        DataSet.GetRasterBand(i+1).WriteArray(data[i])
        DataSet.GetRasterBand(i+1).SetNoDataValue(NoValue)
else:
    DataSet.GetRasterBand(i+1).WriteArray(data[0][0])
    DataSet.GetRasterBand(i+1).SetNoDataValue(NoValue)
DataSet.FlushCache()
#%%
NetCDFtoRaster(FileName, SaveTo, VarName)
#%%
import netCDF4
import numpy as np
from osgeo import gdal
from osgeo import osr

#Reading in data from files and extracting said data
ncfile = netCDF4.Dataset(FileName, 'r')
dataw = ncfile.variables["dataw"][:]
lat = ncfile.variables["Latitude"][:]
long = ncfile.variables["Longitude"][:]


n = len(dataw)
x = np.zeros((n,3), float)

x[:,0] = long[:]
x[:,1] = lat[:]
x[:,2] = dataw[:]

nx = len(long)
ny = len(lat)
xmin, ymin, xmax, ymax = [long.min(), lat.min(), long.max(), lat.max()]
xres = (xmax - xmin) / float(nx)
yres = (ymax - ymin) / float(ny)
geotransform = (xmin, xres, 0, ymax, 0, -yres)

#Creates 1 raster band file
dst_ds = gdal.GetDriverByName('GTiff').Create('myGeoTIFF.tif', ny, nx, 1, gdal.GDT_Float32)

dst_ds.SetGeoTransform(geotransform)    # specify coords
srs = osr.SpatialReference()            # establish encoding
srs.ImportFromEPSG(3857)                # WGS84 lat/long
dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
dst_ds.GetRasterBand(1).WriteArray(x)   # write r-band to the raster
dst_ds.FlushCache()                     # write to disk
dst_ds = None                           # save, close

#%% plot

src = rasterio.open(SaveTo)
fig = plt.figure(figsize=(12,8))
im = plt.imshow(src.read(1)/100.0, cmap="gist_rainbow")
plt.title("Monthly mean sea level pressure")
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
plt.colorbar(im, cax=cax)
plt.tight_layout()
plt.show()
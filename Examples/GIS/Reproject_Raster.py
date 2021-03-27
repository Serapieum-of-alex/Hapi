# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:04:28 2021

@author: mofarrag
"""

import os
# os.chdir("F:/01Algorithms/Hydrology/HAPI/Examples")

from Hapi.raster import Raster
import gdal
import osr

RasterApath = "data/GIS/DEM5km_Rhine_burned_acc.tif"
RasterBpath = "data/GIS/MSWEP_1979010100.tif"
# RasterBpath = "F:/01Algorithms/Hydrology/HAPI/Examples/data/GIS/MSWEP_4746epsg.tif"
SaveTo = "data/GIS/MSWEP_1979010100_reprojected.tif"

RasterA = gdal.Open(RasterApath)
RasterB = gdal.Open(RasterBpath)


# we need number of rows and cols from src A and data from src B to store both in dst
RasterA_proj=RasterA.GetProjection()
RasterA_epsg=osr.SpatialReference(wkt=RasterA_proj)


RasterB_reprojected = Raster.ProjectRaster(RasterB,int(RasterA_epsg.GetAttrValue('AUTHORITY',1)),
                                           resample_technique="cubic", Option=1)

# GET THE GEOTRANSFORM
RasterB_gt=RasterB.GetGeoTransform()
# GET NUMBER OF columns
RasterB_x=RasterB.RasterXSize
# get number of rows
RasterB_y=RasterB.RasterYSize
# we need number of rows and cols from src A and data from src B to store both in dst
RasterB_proj=RasterB.GetProjection()
RasterB_epsg=osr.SpatialReference(wkt=RasterB_proj)


# GET THE GEOTRANSFORM
RasterB_reprojected_gt=RasterB_reprojected.GetGeoTransform()
# GET NUMBER OF columns
RasterB_reprojected_x=RasterB_reprojected.RasterXSize
# get number of rows
RasterB_reprojected_y=RasterB_reprojected.RasterYSize
# we need number of rows and cols from src A and data from src B to store both in dst
RasterB_reprojected_proj=RasterB_reprojected.GetProjection()
RasterB_reprojected_epsg=osr.SpatialReference(wkt=RasterB_reprojected_proj)

RasterB_reprojected_array = RasterB_reprojected.ReadAsArray()

#%% save the raster
Raster.SaveRaster(RasterB_reprojected,SaveTo)
#%% 
RasterB_reprojected = Raster.ProjectRaster(RasterB,int(RasterA_epsg.GetAttrValue('AUTHORITY',1)),
                                           resample_technique="cubic", Option=2)
SaveTo = "data/GIS/MSWEP_1979010100_reprojected2.tif"
Raster.SaveRaster(RasterB_reprojected,SaveTo)
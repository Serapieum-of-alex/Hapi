# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 21:44:12 2021

@author: mofarrag
"""
import os
Comp = "F:/Users/mofarrag/"
os.chdir(Comp + "/Coello/HAPI/Data")
from osgeo import gdal
# from gdalconst import GA_ReadOnly
import osr
from osgeo import gdalconst
from Hapi.raster import Raster

SourceRasterPath ="00inputs/GIS/4000/acc4000.tif"
RasterTobeClippedPath = Comp + "/Documents/01Algorithms/HAPI/Hapi/Parameters/01/Par_BETA.tif"
output ='F:/Users/mofarrag/coello/Hapi/Data/output.tif'
#%%

Raster.ClipRasterWithRaster(RasterTobeClippedPath, SourceRasterPath, output, Save=True)
#%%

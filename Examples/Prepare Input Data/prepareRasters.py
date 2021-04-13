# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 19:09:20 2021

@author: mofarrag
"""
import os
Comp = "F:/Users/mofarrag/"
os.chdir(Comp + "/Coello/HAPI/Data")
from Hapi.raster import Raster
import gdal
import numpy as np

dem_path="00inputs/GIS/4000/acc4000.tif"
# SaveTo = "F:/Users/mofarrag/coello/Hapi/Data/00inputs/Basic_inputs/default parameters/01/"
SaveTo = "F:/Users/mofarrag/coello/Hapi/Model/results/parameters/4000/lumped/2021-03-30/rasters/"
#%%
'craeate a raster typicall to the DEM and fill it with 1'
K = 1
src = gdal.Open(dem_path)

Raster.RasterFill(src, K, SaveTo+'11_K_muskingum.tif')
#%%
X = 0.2

Raster.RasterFill(src, X, SaveTo+'12_X_muskingum.tif')

#%%


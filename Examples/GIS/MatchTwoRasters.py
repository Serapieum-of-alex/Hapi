# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:43:22 2019

@author: mofarrag

match two rasters
"""
#%links
from IPython import get_ipython   # to reset the variable explorer each time
get_ipython().magic('reset -f')
import os
os.chdir("F:/01Algorithms/HAPI/Examples/GIS")


#%library
import numpy as np
import gdal
import datetime as dt
#import pandas as pd
import Hapi.raster as GIS
#import matplotlib.pyplot as plt
#%% inputs
RasterAPath = "Inputs/NewDEM.tif"
RasterBpath = "Inputs/SWIM_sub_4647.tif"



#%%
"""
Read the Input rasters

"""
# the source raster is of the ASCII format
src = gdal.Open(RasterAPath)
#src_Array = src.ReadAsArray()

# read destination array
dst = gdal.Open(RasterBpath)
#Dst_Array = dst.ReadAsArray()
NewRasterB = GIS.MatchRasterAlignment(src,dst)

GIS.SaveRaster(NewRasterB,"Inputs/NewSWIMSubs.tif")

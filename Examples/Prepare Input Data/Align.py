# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 03:34:11 2021

@author: mofarrag
"""
import os
# Comp = "F:/02Case studies/"
Comp = "F:/Users/mofarrag/"
os.chdir(Comp + "Coello/HAPI/Data")
#%library
import numpy as np
# functions
from Hapi.raster import Raster
from Hapi.inputs import Inputs
import gdal

"""
this function prepare downloaded raster data to have the same align and
nodatavalue from a GIS raster (DEM, flow accumulation, flow direction raster)
and return a folder with the output rasters with a name “New_Rasters”
"""
dem_path = "00inputs/GIS/4000/acc4000.tif"
# outputpath = "meteodata/4000/calib/MSWEP-prec/"
# prec
# prec_in_path="D:/MSWEP/Daily_010deg/04tiff/"
prec_in_path = "F:/Users/mofarrag/New folder (2)/"
outputpath = "meteodata/4000/calib/CPC-NOAA/"
Inputs.PrepareInputs(dem_path,prec_in_path,outputpath+"prec")
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 03:34:11 2021

@author: mofarrag

Make sure the working directory is set to the examples folder in the Hapi repo"
currunt_work_directory = Hapi/Example

"""
from Hapi.inputs import Inputs


"""
this function prepare downloaded raster data to have the same align and
nodatavalue from a GIS raster (DEM, flow accumulation, flow direction raster)
and return a folder with the output rasters with a name “New_Rasters”
"""
dem_path = "Data/GIS/Hapi_GIS_Data/acc4000.tif"
outputpath = "data/PrepareMeteodata/meteodata_prepared/prec"

# prec
prec_in_path = "data/PrepareMeteodata/raw_data/prec/"
Inputs.PrepareInputs(dem_path, prec_in_path, outputpath)

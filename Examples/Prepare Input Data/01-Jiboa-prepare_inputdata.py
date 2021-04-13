# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 23:51:26 2018

@author: Mostafa
"""
#%library
import numpy as np
# functions
from Hapi.raster import Raster
from Hapi.inputs import Inputs

import gdal
import os
os.chdir("F:/02Case studies/El Salvador/Examples")
"""
this function prepare downloaded raster data to have the same align and
nodatavalue from a GIS raster (DEM, flow accumulation, flow direction raster)
and return a folder with the output rasters with a name “New_Rasters”
"""
dem_path="Data/GIS/4000/acc4000.tif"

outputpath="Data/meteodata_prepared/"
# prec
prec_in_path="Data/meteodata/4000/complete_dataset/prec/"
Inputs.PrepareInputs(dem_path,prec_in_path,outputpath+"prec")

# evap
evap_in_path="Data/meteodata/4000/complete_dataset/evap/"
# Inputs.PrepareInputs(dem_path,evap_in_path,outputpath+"evap")
# temp
temp_in_path="Data/meteodata/4000/complete_dataset/temp/"
# Inputs.PrepareInputs(dem_path,temp_in_path,outputpath+"temp")

"""
in case you want to manipulate the value in all the rasters of one of the inputs
for example evapotranspiration values in rasters downloaded from ECMWF are -ve
and to change it to +ve in all rasters or if you want to operate any kind of function
in all input rasters that are in the same folder FolderCalculator can do this task

"How can evaporation have both positive and negative values?
Evaporation is normally negative due to the convention for fluxes.
The meteorological convention for all vertical fluxes is that downwards is positive.
Positive evaporation represents condensation'.
Link: https://confluence.ecmwf.int/pages/viewpage.action?pageId=111155327
"""
evap_out_path="Data/meteodata_prepared/evap/"

# define your function
# this function is going to take the absolute value of the values in the raster
# through MapAlgebra function then save the new raster to a given path with the same names
def function(args):
    # argument a list of two components
    # first argument is the raster object [gdal object]
    A = args[0]
    # second argument is the path to save the resulted raster
    path = args[1]
    func=np.abs
    # first function
    B=Raster.MapAlgebra(A,func)
    Raster.SaveRaster(B,path)

folder_path = evap_out_path
new_folder_path="data/meteodata_prepared/new_evap/"
# Raster.FolderCalculator(folder_path,new_folder_path,function)

"""
in order to run the model all inputs have to have the same number of rows and columns
for this purpose MatchRasterAlignment function was made to resample, change the coordinate
system of the second raster and give it the same alignment like a source raster (DEM raster)
"""

soil_path="Data/GIS/soil/4000/soil_raster.tif"
DEM=gdal.Open(dem_path)
dem_A=DEM.ReadAsArray()
soil=gdal.Open(soil_path)
soil_A=soil.ReadAsArray()

# align
aligned_soil=Raster.MatchRasterAlignment(DEM,soil)

# to check alignment of DEM raster compared to aligned_soil_A raster
aligned_soil_A=aligned_soil.ReadAsArray()

# nodatavalue is still different and some cells are no data value in the soil type raster but it is not in the dem raster
# to match use Match MatchNoDataValue
# match
dst_Aligned_M=Raster.MatchNoDataValue(DEM,aligned_soil)
dst_Aligned_M_A=dst_Aligned_M.ReadAsArray()

# save the new raster
Raster.SaveRaster(dst_Aligned_M,"Data/GIS/soil/4000/soil_type.tif")

#Raster.SaveRaster(dst_Aligned_M,"00inputs/GIS/4000/soil_typeِِA.tif")

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 23:51:26 2018

@author: Mostafa
"""
#%links
from IPython import get_ipython   # to reset the variable explorer each time
get_ipython().magic('reset -f')
import os
os.chdir("F:/02Case studies/Coello/HAPI/Data")
import sys
# sys.path.append("C:/Users/Mostafa/Desktop/My Files/thesis/My Thesis/Data_and_Models/Interface/Distributed_Hydrological_model/HBV_distributed/function")

#%library
import numpy as np
# functions
import gdal
import Hapi.raster as GIS
import Hapi.inputs as Inputs
#%%
### 4 km
# dem_path="01GIS/inputs/4000/acc4000.tif"
dem_path="00inputs/GIS/4000/acc4000.tif"
outputpath="00inputs/meteodata/4000/"
# prec
prec_in_path="02Precipitation/CHIRPS/Daily/"
# Inputs.PrepareInputs(dem_path,prec_in_path,outputpath+"prec")
# evap
evap_in_path="03Weather_Data/evap/"
#Inputs.PrepareInputs(dem_path,evap_in_path,outputpath+"evap")
# temp
temp_in_path="03Weather_Data/temp/"
#Inputs.PrepareInputs(dem_path,temp_in_path,outputpath+"temp")

#%%
evap_out_path="03Weather_Data/new/4km_f/evap/"

def function(args):
    A = args[0]
    func=np.abs
    path = args[1]

    B=GIS.MapAlgebra(A,func)
    GIS.SaveRaster(B,path)

#%%
#files_list=os.listdir(folder_path)
folder_path = evap_out_path
new_folder_path="03Weather_Data/new/4km_f/new_evap/"

#B=gdal.Open(folder_path+files_list[0])
#args=[B,func,new_folder_path+files_list[0]]

#GIS.FolderCalculator(folder_path,new_folder_path,function)

#%% soil raster
dem_path="01GIS/inputs/4000/acc4000.tif"
soil_path="01GIS/soil_type/TOLIMA_SUELOS_VF/soil_raster.tif"
DEM=gdal.Open(dem_path)
dem_A=DEM.ReadAsArray()

# dst
dst=gdal.Open(soil_path)
dst_A=dst.ReadAsArray()

# align
dst_Aligned=GIS.MatchRasterAlignment(DEM,dst)
dst_Aligned_A=dst_Aligned.ReadAsArray()
#noval_Aligned=np.float32(dst_Aligned.GetRasterBand(1).GetNoDataValue())
# match
dst_Aligned_M=GIS.MatchNoDataValue(DEM,dst_Aligned)
dst_Aligned_M_A=dst_Aligned_M.ReadAsArray()

GIS.SaveRaster(dst_Aligned_M,"00inputs/GIS/4000/soil_typeِِA.tif")

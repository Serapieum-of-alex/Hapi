# -*- coding: utf-8 -*-
"""
this code creates the max and min values of the parameters from the 10 scenarios
of the HBV-SIMREG

@author: mofarrag
"""
from IPython import get_ipython
get_ipython().magic("reset -f")
# Comp = "F:/"
Comp = "F:/Users/mofarrag/Documents/"
# import os
import gdal
# import osr
# import numpy as np
# import gdalconst
from Hapi.raster import Raster as R

ParamList = ["01_tt", "02_rfcf", "03_sfcf","04_cfmax","05_cwh","06_cfr","07_fc","08_beta", "09_etf",
                 "10_lp","11_k0","12_k1","13_k2","14_uzl","15_perc", "16_maxbas", "17_K_muskingum", 
                 "18_x_muskingum"]
SaveTo = Comp + "01Algorithms/HAPI/Hapi/Parameters/" 
# par = "UZL"
for i in range(len(ParamList)):
    Path = list()
    for j in range(0,10):
        if j < 9 : 
            folder = "0" + str(j+1)
        else:
            folder = str(j+1)
            
        Path.append(Comp + "01Algorithms/HAPI/Hapi/Parameters/" + folder + "/" + ParamList[i]+ ".tif") 
        
    parameters = R.ReadRastersFolder(Path, WithOrder=False)
    MaxValue = parameters.max(axis=2)
    MinValue = parameters.min(axis=2)
    MeanValue = parameters.mean(axis=2)
    
    # Path1 = path + "/" + ParamList[i] + "-1.tif"
    src = gdal.Open(Path[0])
    
    Saveto1 = SaveTo + "/max/" + ParamList[i] + ".tif"
    Saveto2 = SaveTo  + "/min/" + ParamList[i] + ".tif"
    Saveto3 = SaveTo  + "/avg/" + ParamList[i] + ".tif"
    
    
    R.RasterLike(src,MaxValue,Saveto1,pixel_type=1)
    R.RasterLike(src,MinValue,Saveto2,pixel_type=1)
    R.RasterLike(src,MeanValue,Saveto3,pixel_type=1)

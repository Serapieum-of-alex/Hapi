# -*- coding: utf-8 -*-
"""
this code creates the max and min values of the parameters from the 10 scenarios
of the HBV-SIMREG

@author: mofarrag
"""
from IPython import get_ipython
get_ipython().magic("reset -f")
# import os
import gdal
# import osr
# import numpy as np
# import gdalconst
import Hapi.raster as R

par = "Par_UZL"
path = "F:/01Algorithms/HBV-SIMREG/parameters/" + par
parameters = R.ReadRastersFolder(path, WithOrder=False)
MaxValue = parameters.max(axis=2)
MinValue = parameters.min(axis=2)

Path1 = path + "/" + par + "-1.tif"
Saveto1 = path + "/" + par + "-Max.tif"
Saveto2 = path + "/" + par + "-Min.tif"
src = gdal.Open(Path1)

R.RasterLike(src,MaxValue,Saveto1,pixel_type=1)
R.RasterLike(src,MinValue,Saveto2,pixel_type=1)

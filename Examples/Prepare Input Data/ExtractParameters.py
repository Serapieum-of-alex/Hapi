# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 18:45:56 2021

@author: mofarrag
"""
import os
Comp = "F:/01Algorithms/Hydrology/HAPI/Examples/"
os.chdir(Comp + "/Coello/HAPI/Data")

from Hapi.inputs import Inputs

dem_path = "Data/GIS/Hapi_GIS_Data/acc4000.tif"
outputpath = Comp + "coello/Hapi/Model/results/parameters/00default parameters/mean/"

Inputs.ExtractParameters(dem_path,'avg', AsRaster=True, SaveTo=outputpath)
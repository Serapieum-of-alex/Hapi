# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 18:45:56 2021

@author: mofarrag
"""
import os
Comp = "F:/Users/mofarrag/"
os.chdir(Comp + "/Coello/HAPI/Data")

from Hapi.inputs import Inputs

dem_path = "00inputs/GIS/4000/acc4000.tif"
outputpath = Comp + "coello/Hapi/Model/results/parameters/00default parameters/mean/"

Inputs.ExtractParameters(dem_path,'avg', AsRaster=True, SaveTo=outputpath)
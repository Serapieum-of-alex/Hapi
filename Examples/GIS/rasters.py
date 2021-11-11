# -*- coding: utf-8 -*-
"""
Created on Sun May  9 03:35:25 2021

@author: mofarrag
"""
import os

os.chdir("F:/01Algorithms/Hydrology/HAPI/Examples/")


from Hapi.gis.raster import Raster

Path = "data/GIS/raster-folder/"
F = Raster.ReadRastersFolder(Path, WithOrder=True)

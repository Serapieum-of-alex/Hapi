# -*- coding: utf-8 -*-
"""
Created on Sat May 16 00:02:08 2020

@author: mofarrag
"""
from IPython import get_ipython
get_ipython().magic("reset -f")
# import os
# os.chdir("")
import Hapi.raster as Raster
import geopandas as gpd


Raster_path = "F:/02Case studies/Rhine/base_data/GIS/Layers/DEM/srtm/DEM_Germany.tif"
shapefile_path = "F:/02Case studies/Rhine/base_data/GIS/Layers/DEM/srtm/cropDEM.shp"

shpfile=gpd.read_file(shapefile_path)


output_path = "F:/02Case studies/Rhine/base_data/GIS/Layers/DEM/srtm/DEM_GermanyC.tif"

Raster.Clip(Raster_path,shapefile_path,save=True,output_path=output_path)

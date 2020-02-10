# -*- coding: utf-8 -*-
"""
Created on Tue May 22 09:01:55 2018

@author: Mostafa
"""
# documentation format
__docformat__ = 'restructuredtext'

# Let users know if they're missing any of our hard dependencies
hard_dependencies = ("numpy", "pandas", "gdal")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(dependency)

if missing_dependencies:
    raise ImportError(
        "Missing required dependencies {0}".format(missing_dependencies))

del hard_dependencies, dependency, missing_dependencies


#import Calibration
import Hapi.DistParameters
###
import Hapi.DistRRM
import Hapi.GISCatchment
import Hapi.GISpy
import Hapi.GISVisual
import Hapi.Inputs
import Hapi.java_functions
import Hapi.PerformanceCriteria
import Hapi.Routing
import Hapi.RUN
import Hapi.StatisticalTools
import Hapi.Wrapper
import Hapi.WeirdFn
import Hapi.HBV
import Hapi.HBV_Lake
import Hapi.HBVLumped
import Hapi.HBVBergestrom92
import Hapi.FloodModel
import Hapi.River

# module level doc-string
__doc__ = """
HAPI - Hydrological library for Python
=====================================================================

**HAPI** is a Python package providing fast and flexible, way to build distributed
hydrological model using lumped conceptual model 

Main Features
-------------
Here are just a few of the things that pandas does well:

  - Easy handling of rasters data downloaded from global data and easy way to
    manipulate the data to arrange it to run the model
  - Easy calibration of the model using Harmony search method and Genetic Algorithms
  - flexible GIS function to process rasters interpolate values and georeference 
   calculated discharge values to the correct place
   
"""

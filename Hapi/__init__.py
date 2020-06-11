# -*- coding: utf-8 -*-
"""
Created on Tue May 22 09:01:55 2018

@author: Mostafa
"""
# documentation format
__author__ = 'Mostafa Farrag'
__version__ = '0.2.0'

__docformat__ = 'restructuredtext'

# Let users know if they're missing any of our hard dependencies
hard_dependencies = () #("numpy", "pandas", "gdal")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(dependency)

if missing_dependencies:
    raise ImportError(
        "Missing required dependencies {0}".format(missing_dependencies))

# del hard_dependencies, dependency, missing_dependencies


#import Calibration
import Hapi.distparameters
import Hapi.distrrm
import Hapi.giscatchment
import Hapi.raster
import Hapi.vector
import Hapi.inputs
import Hapi.java_functions
import Hapi.performancecriteria
import Hapi.routing
import Hapi.run
import Hapi.statisticaltools
import Hapi.wrapper
import Hapi.weirdFn
import Hapi.hbv
import Hapi.hbv_lake
import Hapi.hbvlumped
import Hapi.hbv_bergestrom92
import Hapi.riminputs
import Hapi.event
import Hapi.river
import Hapi.visualizer
import Hapi.crosssection
import Hapi.rimcalibration
# module level doc-string
__doc__ = """
Hapi - Hydrological library for Python
=====================================================================

**Hapi** is a Python package providing fast and flexible, way to build distributed
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
